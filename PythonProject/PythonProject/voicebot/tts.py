"""Text-to-speech helper.

Primary: Windows SAPI5 via win32com (most reliable on Windows).
Fallback 1: pyttsx3 (cross-platform).
Fallback 2: print() if both are unavailable.
"""
from typing import Optional
import logging

log = logging.getLogger(__name__)


class TTS:
    def __init__(self, voice_name: Optional[str] = None, rate: Optional[int] = None):
        self._engine_type = None
        self._engine = None
        self._enabled = False

        # --- Try Windows SAPI5 via win32com first (most reliable on Windows) ---
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")

            # Optionally select voice
            if voice_name:
                voices = speaker.GetVoices()
                for i in range(voices.Count):
                    v = voices.Item(i)
                    if voice_name.lower() in v.GetDescription().lower():
                        speaker.Voice = v
                        break

            # Optionally set rate (-10 slow … 10 fast; map wpm roughly)
            if rate is not None:
                # pyttsx3 default ~200 wpm → SAPI rate 0; scale roughly
                sapi_rate = max(-10, min(10, int((rate - 200) / 30)))
                speaker.Rate = sapi_rate

            self._engine = speaker
            self._engine_type = 'sapi'
            self._enabled = True
            log.debug('TTS: using Windows SAPI5 via win32com')

        except Exception as e:
            log.debug('win32com SAPI5 unavailable: %s', e)

        # --- Fallback: pyttsx3 ---
        if not self._enabled:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                if voice_name:
                    voices = engine.getProperty('voices')
                    for v in voices:
                        if voice_name.lower() in v.name.lower():
                            engine.setProperty('voice', v.id)
                            break
                if rate is not None:
                    engine.setProperty('rate', rate)
                self._engine = engine
                self._engine_type = 'pyttsx3'
                self._enabled = True
                log.debug('TTS: using pyttsx3')
            except Exception as e:
                log.debug('pyttsx3 unavailable: %s', e)
                self._engine = None
                self._enabled = False

    def speak(self, text: str) -> bool:
        """Speak the provided text. Returns True if audio was produced, False otherwise."""
        if not text:
            return False
        if not self._enabled or self._engine is None:
            return False

        try:
            if self._engine_type == 'sapi':
                # SVSFlagsAsync=1 would be async; 0 = synchronous (blocks until done)
                self._engine.Speak(text, 0)
                return True
            elif self._engine_type == 'pyttsx3':
                self._engine.say(text)
                self._engine.runAndWait()
                return True
        except Exception as e:
            log.warning('TTS engine error (%s): %s', self._engine_type, e)
            return False

        return False
