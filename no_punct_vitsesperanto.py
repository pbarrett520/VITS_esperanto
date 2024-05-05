import os

from trainer import Trainer,TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.models.vits import VitsCharacters
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))

# specify which characters are going to be used in this model
characters = CharactersConfig(
    characters="KoNtndOzŜheĉĴmsuPEĜŝIDZaĈjJSŭFlkTĥULgMHvĤpĝbRBGrCifĵcVA",
    punctuations=" ",
    characters_class="TTS.tts.models.vits.VitsCharacters",
    pad="<PAD>",
)
# specify that 'no_punct.csv' is the the meta file, not metadata.csv
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="no_punct.csv",
    path=os.path.join(output_path,"/home/u6/pbarrett520/vits")
)
# vanilla audi config
audio_config = BaseAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=True,
    trim_db=45,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
)
# vanilla VITS cofig
config = VitsConfig(
    characters=characters,
    audio=audio_config,
    run_name="vits_esperanto_no_punct",
    batch_size=16,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=0,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=350,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    test_sentences=[
        ["Mi sentas la mankon de vi."],
        ["Kiel vi povas fari tion al mi?"],
        ["Mi ŝatasrenkonti novajn homojn."]
    ],
)

#INITIALIZE AUDIO PROCESSOR
ap = AudioProcessor.init_from_config(config)

#INITIALIZE TOKENIZER
tokenizer,config = TTSTokenizer.init_from_config(config)

#LOAD DATA SAMPLES
train_samples,eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True
)

#init model
model = Vits(config,ap,tokenizer,speaker_manager=None)

#init trainer
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()

