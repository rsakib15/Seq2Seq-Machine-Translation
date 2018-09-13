
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TextDataset,tensorFromSentence
from model.seq2seq import AttnDecoderRNN, DecoderRNN, EncoderRNN

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
use_attn = True
use_cuda = torch.cuda.is_available()
lang_dataset = TextDataset()


lang_dataloader = DataLoader(lang_dataset, shuffle=True)
input_size = lang_dataset.input_lang_words
hidden_size = 256
output_size = lang_dataset.output_lang_words
total_epoch = 20

encoder = EncoderRNN(lang_dataset.input_lang_words, hidden_size, n_layers=2)
decoder = DecoderRNN(lang_dataset.output_lang_words, output_size, n_layers=2)
attn_decoder = AttnDecoderRNN(hidden_size, output_size, n_layers=2)


def evaluate(sentence, max_length=MAX_LENGTH):
    input_tensor = tensorFromSentence(lang_dataset.input_lang, sentence)
    input_length = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]





    # Run through decoder
    print(lang_dataset.input_lang.index2word)
    print(lang_dataset.output_lang.index2word)
    return decoded_words


def main():
    user_input = 'how are you'
    sentence = user_input
    output_words = evaluate(sentence)
    output_sentence = ' '.join(output_words)
    print("Sentence: {}\nTranslated Sentence: {}".format(user_input, output_sentence))


if __name__ == "__main__":
    main()