package openai //nolint:testpackage // testing private field

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"testing"

	utils "github.com/meguminnnnnnnnn/go-openai/internal"
	"github.com/meguminnnnnnnnn/go-openai/internal/test"
	"github.com/meguminnnnnnnnn/go-openai/internal/test/checks"
)

var errTestUnmarshalerFailed = errors.New("test unmarshaler failed")

type failingUnMarshaller struct{}

func (*failingUnMarshaller) Unmarshal(_ []byte, _ any) error {
	return errTestUnmarshalerFailed
}

func TestStreamReaderReturnsUnmarshalerErrors(t *testing.T) {
	stream := &streamReader[ChatCompletionStreamResponse]{
		errAccumulator: utils.NewErrorAccumulator(),
		unmarshaler:    &failingUnMarshaller{},
	}

	respErr := stream.unmarshalError()
	if respErr != nil {
		t.Fatalf("Did not return nil with empty buffer: %v", respErr)
	}

	err := stream.errAccumulator.Write([]byte("{"))
	if err != nil {
		t.Fatalf("%+v", err)
	}

	respErr = stream.unmarshalError()
	if respErr != nil {
		t.Fatalf("Did not return nil when unmarshaler failed: %v", respErr)
	}
}

func TestStreamReaderReturnsErrTooManyEmptyStreamMessages(t *testing.T) {
	stream := &streamReader[ChatCompletionStreamResponse]{
		emptyMessagesLimit: 3,
		reader:             bufio.NewReader(bytes.NewReader([]byte("\n\n\n\n"))),
		errAccumulator:     utils.NewErrorAccumulator(),
		unmarshaler:        &utils.JSONUnmarshaler{},
	}
	_, err := stream.Recv()
	checks.ErrorIs(t, err, ErrTooManyEmptyStreamMessages, "Did not return error when recv failed", err.Error())
}

func TestStreamReaderReturnsErrTestErrorAccumulatorWriteFailed(t *testing.T) {
	stream := &streamReader[ChatCompletionStreamResponse]{
		reader: bufio.NewReader(bytes.NewReader([]byte("\n"))),
		errAccumulator: &utils.DefaultErrorAccumulator{
			Buffer: &test.FailingErrorBuffer{},
		},
		unmarshaler: &utils.JSONUnmarshaler{},
	}
	_, err := stream.Recv()
	checks.ErrorIs(t, err, test.ErrTestErrorAccumulatorWriteFailed, "Did not return error when write failed", err.Error())
}

func TestStreamReaderRecvRaw(t *testing.T) {
	stream := &streamReader[ChatCompletionStreamResponse]{
		reader: bufio.NewReader(bytes.NewReader([]byte("data: {\"key\": \"value\"}\n"))),
	}
	rawLine, err := stream.RecvRaw()
	if err != nil {
		t.Fatalf("Did not return raw line: %v", err)
	}
	if !bytes.Equal(rawLine, []byte("{\"key\": \"value\"}")) {
		t.Fatalf("Did not return raw line: %v", string(rawLine))
	}
}

func TestStreamReaderMultilineDataEvent(t *testing.T) {
	// A single SSE event whose JSON payload is split across two "data:"
	// lines must be joined with "\n" and dispatched as one event.
	body := "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\n" +
		"data: \"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n" +
		"\n" +
		"data: [DONE]\n"

	stream := &streamReader[ChatCompletionStreamResponse]{
		reader:         bufio.NewReader(bytes.NewReader([]byte(body))),
		errAccumulator: utils.NewErrorAccumulator(),
		unmarshaler:    &utils.JSONUnmarshaler{},
	}

	rawLine, err := stream.RecvRaw()
	if err != nil {
		t.Fatalf("Did not return raw line: %v", err)
	}
	want := "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\n" +
		"\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}"
	if !bytes.Equal(rawLine, []byte(want)) {
		t.Fatalf("Did not join multiline data payload, got: %q", string(rawLine))
	}

	if _, err = stream.RecvRaw(); !errors.Is(err, io.EOF) {
		t.Fatalf("Expected io.EOF after [DONE], got: %v", err)
	}
}

func TestStreamReaderRecvMultilineDataEvent(t *testing.T) {
	body := "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\n" +
		"data: \"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n" +
		"\n" +
		"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n" +
		"\n" +
		"data: [DONE]\n\n"

	stream := &streamReader[ChatCompletionStreamResponse]{
		reader:         bufio.NewReader(bytes.NewReader([]byte(body))),
		errAccumulator: utils.NewErrorAccumulator(),
		unmarshaler:    &utils.JSONUnmarshaler{},
	}

	resp, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv failed on multiline event: %v", err)
	}
	if len(resp.Choices) != 1 || resp.Choices[0].Delta.Content != "hello" {
		t.Fatalf("Unexpected response: %+v", resp)
	}

	resp, err = stream.Recv()
	if err != nil {
		t.Fatalf("Recv failed on single-line event: %v", err)
	}
	if len(resp.Choices) != 1 || string(resp.Choices[0].FinishReason) != "stop" {
		t.Fatalf("Unexpected response: %+v", resp)
	}

	if _, err = stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("Expected io.EOF after [DONE], got: %v", err)
	}
}
