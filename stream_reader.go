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
	headerData = regexp.MustCompile(`^data:\s*`)
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
	var emptyMessagesCount uint
	var eventData bytes.Buffer

	for {
		rawLine, readErr := stream.reader.ReadBytes('\n')
		noSpaceLine := bytes.TrimSpace(rawLine)

		if len(noSpaceLine) == 0 {
			if eventData.Len() > 0 {
				return stream.processEvent(eventData.Bytes())
			}
			if err := stream.errAccumulator.Write(noSpaceLine); err != nil {
				return nil, err
			}
		} else if headerData.Match(noSpaceLine) {
			dataLine := headerData.ReplaceAll(noSpaceLine, nil)
			if eventData.Len() > 0 {
				eventData.WriteByte('\n')
			}
			eventData.Write(dataLine)
		} else {
			writeErr := stream.errAccumulator.Write(noSpaceLine)
			if writeErr != nil {
				return nil, writeErr
			}
		}

		if readErr != nil {
			if errors.Is(readErr, io.EOF) && eventData.Len() > 0 {
				return stream.processEvent(eventData.Bytes())
			}
			respErr := stream.unmarshalError()
			if respErr != nil {
				return nil, fmt.Errorf("error, %w", respErr.Error)
			}
			return nil, readErr
		}

		if len(noSpaceLine) == 0 || !headerData.Match(noSpaceLine) {
			emptyMessagesCount++
			if emptyMessagesCount > stream.emptyMessagesLimit {
				return nil, ErrTooManyEmptyStreamMessages
			}
		}
	}
}

func (stream *streamReader[T]) processEvent(data []byte) ([]byte, error) {
	if string(data) == "[DONE]" {
		stream.isFinished = true
		return nil, io.EOF
	}
	if bytes.HasPrefix(data, []byte(`{"error":`)) {
		if err := stream.errAccumulator.Write(data); err != nil {
			return nil, err
		}
		if respErr := stream.unmarshalError(); respErr != nil {
			return nil, fmt.Errorf("error, %w", respErr.Error)
		}
	}
	return data, nil
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
