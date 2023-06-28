import asyncio
import json
import threading
import pyaudio
import paddlehub
import websockets
from loguru import logger

class ASRWsAudioHandler(threading.Thread):
    def __init__(self, server="ws://127.0.0.1:8091/paddlespeech/asr/streaming"):
        threading.Thread.__init__(self)
        self.server = server
        self.chunk_size = 2048
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_running = True
        self.record_chunks = []
        self.punc_model = paddlehub.Module(name='auto_punc')

    def start_record(self):
        logger.info("开始录音")
        threading._start_new_thread(self.do_record, ())

    def do_record(self):
        self.record_running = True
        self.record_chunks = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk_size)
        while self.record_running:
            self.record_chunks.append(stream.read(self.chunk_size))
        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop_record(self):
        logger.info("结束录音")
        self.record_running = False

    async def run(self):
        name = 'test.wav'
        self.start_record()
        async with websockets.connect(self.server) as ws:
            logger.info('初始化')
            await ws.send(json.dumps(
                {"name": name, "signal": "start", "nbest": 5},
                sort_keys=True, indent=4, separators=(',', ': '))
            )
            msg = await ws.recv()
            logger.info(msg)
            logger.info('识别中')
            try:
                text = ''
                while True:
                    while len(self.record_chunks) > 0:
                        await ws.send(self.record_chunks.pop(0))
                        msg = await ws.recv()
                        msg = json.loads(msg)
                        logger.debug(msg)
                        if msg.get('result', '') != '':
                            text = msg['result']
                        else:
                            if text != '':
                                text = self.punc_model.add_puncs(text)
                                logger.info(f'Recognition Result: {text}')
                                text = ''
            except asyncio.CancelledError:
                await ws.send(json.dumps(
                    {"name": name, "signal": "end", "nbest": 5},
                    sort_keys=True, indent=4, separators=(',', ': ')
                ))
                msg = await ws.recv()
                logger.info(msg)
                self.stop_record()

if __name__ == "__main__":
    logger.info("Start")
    handler = ASRWsAudioHandler()
    loop = asyncio.get_event_loop()
    main_task = asyncio.ensure_future(handler.run())
    try:
        loop.run_until_complete(main_task)
    finally:
        loop.close()
    logger.info("End")
