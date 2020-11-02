# KWS implementation
To check streaming mode of a model, in src/config.ini you should set "mode" in section "common" as "example". In this case when running "python3 KWS.py" it will take predefined long spectrogram with keyword in the middle and will print predicted score of keyword presence along processing the spectrogram.<br>
To check given audio track on presence of keyword, set "mode" to "check", change "../resources/audio_to_check.wav" to your file and execute "python3 KWS.py" This will print model prediction whether keyword is present in audio.
