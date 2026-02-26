"""Command-line interview runner with TTS and microphone cue (beep)."""
from .questions import get_roles, get_questions
from .tts import TTS
from .audio import AudioManager
from .scorer import evaluate
from .logger import save_session, make_session_record
from .config import DEFAULT_SESSION_FILE
import argparse
import sys
import logging
from datetime import datetime

log = logging.getLogger(__name__)


def _beep():
    """Play a short beep to cue the candidate before listening. Windows: winsound; fallback to terminal bell."""
    try:
        # winsound is available on Windows
        import winsound

        # frequency 750 Hz, duration 200 ms
        winsound.Beep(750, 200)
    except Exception:
        # fallback to ASCII bell
        try:
            print("\a", end="", flush=True)
        except Exception:
            # Last-resort: do nothing silently
            pass


def run_interview(role: str, text_only: bool = False, out_file: str = None):
    tts = TTS()
    audio = AudioManager()

    # Report availability so user can see why the program may fall back to text input
    tts_enabled = getattr(tts, "_enabled", False)
    mic_available = audio.has_microphone() if hasattr(audio, "has_microphone") else False
    status_msg = f"TTS {'available' if tts_enabled else 'UNAVAILABLE'}; Microphone {'available' if mic_available else 'UNAVAILABLE'}."
    print(status_msg)
    # Try to announce status via TTS if available
    try:
        if tts_enabled:
            tts.speak(status_msg)
    except Exception:
        # Do not fail the interview if TTS has an unexpected error
        log.debug("TTS status announcement failed", exc_info=True)

    roles = get_roles()
    if role not in roles:
        print(f"Unknown role {role}. Available roles: {roles}")
        role = roles[0]

    questions = get_questions(role)
    results = []

    start_msg = f"Starting interview for role: {role}"
    # Speak if possible, otherwise print
    if not tts.speak(start_msg):
        print(start_msg)

    for idx, q in enumerate(questions, 1):
        qtext = q.get("text", "")
        print(f"\n[QUESTION {idx}/{len(questions)}]")
        print(qtext)
        print()

        # Speak the question (or print fallback)
        qmsg = f"Question {idx}: {qtext}"
        if not tts.speak(qmsg):
            # If TTS unavailable, message was already printed above
            pass

        # Play an audible cue before listening so it feels like a real interview
        _beep()

        if text_only:
            try:
                ans = input("(text input) Your answer: ").strip()
            except EOFError:
                ans = ""
        else:
            # audio.listen will prompt for typed input if mic/recognizer fails
            try:
                ans = audio.listen(prompt="(Listening...)", timeout=8, phrase_time_limit=120)
            except Exception:
                # Fallback to typed input on unexpected error
                log.debug("audio.listen raised unexpected error; falling back to text input", exc_info=True)
                try:
                    ans = input("(text input) Your answer: ").strip()
                except EOFError:
                    ans = ""

        # If answer is empty/blank, allow a couple of retries
        if not ans or not str(ans).strip():
            retries = 2
            for attempt in range(retries):
                reprompt = "No answer detected. Please try again."
                if not tts.speak(reprompt):
                    print(reprompt)
                if text_only:
                    try:
                        new_ans = input("(text input) Your answer: ").strip()
                    except EOFError:
                        new_ans = ""
                else:
                    try:
                        # Play a short beep between retries as well
                        _beep()
                        new_ans = audio.listen(prompt="(Listening...)", timeout=6, phrase_time_limit=60)
                    except Exception:
                        log.debug("Retry audio.listen failed; falling back to typed input", exc_info=True)
                        try:
                            new_ans = input("(text input) Your answer: ").strip()
                        except EOFError:
                            new_ans = ""
                if new_ans and str(new_ans).strip():
                    ans = new_ans
                    break

        ans = ans or ""
        print(f"You said: {ans}")
        # Acknowledge receipt
        if not tts.speak("Thank you"):
            # If TTS fails, we already printed the answer above
            pass

        res = evaluate(ans, q)
        # ensure answer is recorded
        res["answer"] = ans
        results.append(res)

        score = res.get("score", 0)
        max_pts = res.get("max_points", 10)
        short_feedback = res.get("feedback", "")
        print(f"Score: {score}/{max_pts}")
        print(f"Feedback: {short_feedback}")
        # Speak brief feedback
        if not tts.speak(short_feedback):
            # Already printed above
            pass
        print("-" * 70)

    # summarize
    total_score = sum(r.get("score", 0) for r in results)
    total_max = sum(r.get("max_points", 0) for r in results)
    pct = round((total_score / total_max) * 100, 1) if total_max else 0

    summary = f"Interview complete. Score: {total_score}/{total_max} ({pct}%)"
    print(f"\n{summary}\n")
    try:
        tts.speak(summary)
    except Exception:
        # ignore TTS errors while exiting
        log.debug("TTS speak summary failed", exc_info=True)

    record = make_session_record(role, questions, results)
    record["end_timestamp"] = datetime.utcnow().isoformat() + "Z"
    out = out_file or str(DEFAULT_SESSION_FILE)
    try:
        save_session(record, out)
        print(f"Session saved to {out}")
    except Exception as e:
        print(f"Failed to save session: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Voice Interview Bot")
    parser.add_argument("--role", default="AI Engineer", help="Role to interview for")
    parser.add_argument("--text-only", action="store_true", help="Use text input/output only")
    parser.add_argument("--out", help="Output sessions JSON file")
    args = parser.parse_args(argv)
    run_interview(args.role, text_only=args.text_only, out_file=args.out)


if __name__ == "__main__":
    main(sys.argv[1:])