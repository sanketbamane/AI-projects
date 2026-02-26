"""Text-to-speech helper using pyttsx3 with fallback to print."""
from typing import Optional
import logging

log = logging.getLogger(__name__)


class TTS:
    def __init__(self, voice_name: Optional[str] = None, rate: Optional[int] = None):
        try:
            import pyttsx3

            self.engine = pyttsx3.init()
            if voice_name:
                voices = self.engine.getProperty('voices')
                for v in voices:
                    if voice_name.lower() in v.name.lower():
                        self.engine.setProperty('voice', v.id)
                        break
            if rate is not None:
                self.engine.setProperty('rate', rate)
            self._enabled = True
        except Exception as e:
            # pyttsx3 not available; TTS will be disabled. Caller should print text when speak() returns False.
            log.debug('pyttsx3 unavailable: %s', e)
            self.engine = None
            self._enabled = False

    def speak(self, text: str) -> None:
        """Speak the provided text.

        Returns True if TTS engine was used, False if TTS is unavailable or failed. The caller
        can decide to print the text when False.
        """
        if not text:
            return False
        if self._enabled and self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
                return True
            except Exception as e:
                log.warning('TTS engine error: %s', e)
                return False
        return False
