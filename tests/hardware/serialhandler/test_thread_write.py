import threading
from types import SimpleNamespace

from src.hardware.serialhandler.threads.messageconverter import MessageConverter
from src.hardware.serialhandler.threads.threadWrite import threadWrite


class _FakeSerial:
    def __init__(self):
        self.is_open = True
        self.in_waiting = 0
        self.writes = []
        self.flush_calls = 0

    def write(self, payload):
        self.writes.append(payload)
        return len(payload)

    def flush(self):
        self.flush_calls += 1


class _FakeLogFile:
    def __init__(self):
        self.messages = []

    def write(self, message):
        self.messages.append(message)


def _make_writer(serial_con):
    writer = threadWrite.__new__(threadWrite)
    writer.motor_output = "serial"
    writer._zmq_sock = None
    writer.messageConverter = MessageConverter()
    writer.process = SimpleNamespace(
        serialCon=serial_con,
        serialConnected=True,
        serialDevice="/dev/ttyACM0",
        serialLock=threading.Lock(),
    )
    writer.logFile = _FakeLogFile()
    writer.last_blocked_reason = None
    return writer


def test_send_to_serial_does_not_flush_cdc_acm_descriptor():
    serial_con = _FakeSerial()
    writer = _make_writer(serial_con)

    sent, raw_command = writer.send_to_serial({"action": "kl", "mode": 30})

    assert sent is True
    assert raw_command == "#kl:30;;\r\n"
    assert serial_con.writes == [b"#kl:30;;\r\n"]
    assert writer.logFile.messages == ["#kl:30;;\r\n"]
    assert serial_con.flush_calls == 0


def test_send_to_serial_formats_speed_reset_command():
    serial_con = _FakeSerial()
    writer = _make_writer(serial_con)

    sent, raw_command = writer.send_to_serial({"action": "speed", "speed": 0})

    assert sent is True
    assert raw_command == "#speed:0;;\r\n"
    assert serial_con.writes == [b"#speed:0;;\r\n"]


def test_send_to_serial_formats_odometer_reset_command():
    serial_con = _FakeSerial()
    writer = _make_writer(serial_con)

    sent, raw_command = writer.send_to_serial({"action": "odoreset", "request": 1})

    assert sent is True
    assert raw_command == "#odoreset:1;;\r\n"
    assert serial_con.writes == [b"#odoreset:1;;\r\n"]
