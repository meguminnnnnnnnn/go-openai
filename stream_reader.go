package openai

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"

	utils "github.com/meguminnnnnnnnn/go-openai/internal"
)

var (
	headerData  = regexp.MustCompile(`^data:\s*`)
	errorPrefix = regexp.MustCompile(`^data:\s*{"error":`)
)

type streamable interface {
	ChatCompletionStreamResponse | CompletionResponse
}

type streamReader[T streamable] struct {
	emptyMessagesLimit uint
	isFinished         bool

	reader         *bufio.Reader
	response       *http.Response
	errAccumulator utils.ErrorAccumulator
	unmarshaler    utils.Unmarshaler

	// dataBuf assembles the payload of the SSE event currently being read.
	// Per the SSE spec an event may span multiple "data:" lines, which are
	// joined with "\n" and dispatched on the terminating blank line (or EOF).
	dataBuf []byte

	httpHeader
}

func (stream *streamReader[T]) Recv() (response T, err error) {
	rawLine, err := stream.RecvRaw()
	if err != nil {
		return
	}

	err = stream.unmarshaler.Unmarshal(rawLine, &response)
	if err != nil {
		return
	}
	return response, nil
}

func (stream *streamReader[T]) RecvRaw() ([]byte, error) {
	if stream.isFinished {
		return nil, io.EOF
	}

	return stream.processLines()
}

//nolint:gocognit
func (stream *streamReader[T]) processLines() ([]byte, error) {
	var (
		emptyMessagesCount uint
		hasErrorPrefix     bool
	)

	for {
		rawLine, readErr := stream.reader.ReadBytes('\n')
		if readErr != nil {
			// Per the SSE spec, an event still pending when the stream is
			// closed must be dispatched before EOF is reported.
			if errors.Is(readErr, io.EOF) && len(stream.dataBuf) > 0 {
				event := stream.dataBuf
				stream.dataBuf = nil
				return event, nil
			}
			respErr := stream.unmarshalError()
			if respErr != nil {
				return nil, fmt.Errorf("error, %w", respErr.Error)
			}
			return nil, readErr
		}
		if hasErrorPrefix {
			respErr := stream.unmarshalError()
			if respErr != nil {
				return nil, fmt.Errorf("error, %w", respErr.Error)
			}
			return nil, readErr
		}

		noSpaceLine := bytes.TrimSpace(rawLine)
		if errorPrefix.Match(noSpaceLine) {
			hasErrorPrefix = true
		}
		if !headerData.Match(noSpaceLine) || hasErrorPrefix {
			if !hasErrorPrefix && len(stream.dataBuf) > 0 {
				// Blank line terminates the event being assembled: dispatch it.
				event := stream.dataBuf
				stream.dataBuf = nil
				return event, nil
			}
			if hasErrorPrefix {
				noSpaceLine = headerData.ReplaceAll(noSpaceLine, nil)
			}
			writeErr := stream.errAccumulator.Write(noSpaceLine)
			if writeErr != nil {
				return nil, writeErr
			}
			emptyMessagesCount++
			if emptyMessagesCount > stream.emptyMessagesLimit {
				return nil, ErrTooManyEmptyStreamMessages
			}

			continue
		}

		noPrefixLine := headerData.ReplaceAll(noSpaceLine, nil)
		if string(noPrefixLine) == "[DONE]" {
			stream.isFinished = true
			return nil, io.EOF
		}

		// Accumulate the payload line; consecutive "data:" lines of the same
		// event are joined with "\n" and returned together at the boundary.
		if len(stream.dataBuf) > 0 {
			stream.dataBuf = append(stream.dataBuf, '\n')
		}
		stream.dataBuf = append(stream.dataBuf, noPrefixLine...)
	}
}

func (stream *streamReader[T]) unmarshalError() (errResp *ErrorResponse) {
	errBytes := stream.errAccumulator.Bytes()
	if len(errBytes) == 0 {
		return
	}

	err := stream.unmarshaler.Unmarshal(errBytes, &errResp)
	if err != nil {
		errResp = nil
	}

	return
}

func (stream *streamReader[T]) Close() error {
	return stream.response.Body.Close()
}
