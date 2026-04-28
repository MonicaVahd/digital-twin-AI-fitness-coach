from voice_processor import VoiceProcessor
import sounddevice as sd
import soundfile as sf
import time

def test_recording():
    """Test basic audio recording."""
    print("\n=== Testing Audio Recording ===")
    voice_processor = VoiceProcessor()
    
    print("Recording for 5 seconds...")
    audio_data = voice_processor.record_audio()
    
    if audio_data is not None:
        # Save the recording
        sf.write('test_recording.wav', audio_data, voice_processor.sample_rate)
        print("Recording saved to test_recording.wav")
        
        # Play back the recording
        print("Playing back recording...")
        sd.play(audio_data, voice_processor.sample_rate)
        sd.wait()
    else:
        print("Recording failed")

def test_tts():
    """Test text-to-speech using gTTS."""
    print("\n=== Testing Text-to-Speech ===")
    voice_processor = VoiceProcessor()
    
    test_text = "Hello! This is a test of the text to speech system. How are you today?"
    print(f"Converting text to speech: '{test_text}'")
    
    audio_data = voice_processor.text_to_speech(test_text)
    
    if audio_data is not None:
        # Save the audio
        sf.write('test_tts.wav', audio_data, voice_processor.sample_rate)
        print("Audio saved to test_tts.wav")
        
        # Play the audio
        print("Playing generated speech...")
        voice_processor.play_audio(audio_data)
    else:
        print("Text-to-speech failed")

def test_voice_interaction():
    """Test complete voice interaction."""
    print("\n=== Testing Voice Interaction ===")
    voice_processor = VoiceProcessor()
    
    print("Speak for 5 seconds...")
    response = voice_processor.process_voice_input()
    
    if response:
        print(f"Transcribed text: {response.text}")
        if response.audio is not None:
            # Save the response audio
            sf.write('response.wav', response.audio, voice_processor.sample_rate)
            print("Response saved to response.wav")
            
            # Play the response
            print("Playing response...")
            voice_processor.play_audio(response.audio)
    else:
        print("Voice processing failed")

def main():
    """Run all voice tests."""
    print("Starting voice system tests...")
    
    # Test recording
    test_recording()
    time.sleep(1)  # Wait between tests
    
    # Test text-to-speech
    test_tts()
    time.sleep(1)  # Wait between tests
    
    # Test complete interaction
    test_voice_interaction()

if __name__ == "__main__":
    main() 