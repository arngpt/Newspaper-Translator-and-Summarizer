from termcolor import colored
import random
import numpy as np
from collections import Counter
import trax
from trax import layers as tl
from trax.fastmath import numpy as fastnp
from trax.supervised import training
import sys
import os
import textwrap
wrapper = textwrap.TextWrapper(width=70)

np.set_printoptions(threshold=sys.maxsize)
import gradio as gr

model_ft = trax.models.Transformer(
    input_vocab_size=33600,
    d_model=512, d_ff=2048, dropout = 0.1,
    n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
    max_len=2048, mode='eval')

model_ft.init_from_file('model_FT/model.pkl.gz',weights_only=True)

model_gt = trax.models.Transformer(
    input_vocab_size=33600,
    d_model=512, d_ff=2048, dropout = 0.1,
    n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
    max_len=2048, mode='eval')

# Pre-trained Transformer model config in gs://trax-ml/models/translation/ende_wmt32k.gin
# Initialize Transformer using pre-trained weights.
model_gt.init_from_file('gs://trax-ml/models/translation/ende_wmt32k.pkl.gz',
                     weights_only=True)

model_sum = trax.models.TransformerLM(vocab_size=33000, d_model=512, d_ff=2048,n_layers=6, n_heads=8, max_len=4096, dropout=0.1,mode='eval', ff_activation=tl.Relu)

model_sum.init_from_file('model_sum/model.pkl.gz',weights_only=True)

def Translate(input,model=None,VOCAB_FILE=None,VOCAB_DIR=None,speed=None):
    text=input
        
    def tokenize(input_str, vocab_file=None, vocab_dir=None):
        """Encodes a string to an array of integers
        Args:
            input_str (str): human-readable string to encode
            vocab_file (str): filename of the vocabulary text file
            vocab_dir (str): path to the vocabulary file
        Returns:
            numpy.ndarray: tokenized version of the input string
        """
        # Set the encoding of the "end of sentence" as 1
        EOS = 1
        # Use the trax.data.tokenize method. It takes streams and returns streams,
        # we get around it by making a 1-element stream with `iter`.
        inputs = next(trax.data.tokenize(iter([input_str]),
                                        vocab_file=vocab_file, vocab_dir=vocab_dir))
        # Mark the end of the sentence with EOS
        inputs = list(inputs) + [EOS]
        # Adding the batch dimension to the front of the shape
        batch_inputs = np.reshape(np.array(inputs), [1, -1])
        return batch_inputs

    def detokenize(integers, vocab_file=None, vocab_dir=None):
        """Decodes an array of integers to a human readable string
        Args:
            integers (numpy.ndarray): array of integers to decode
            vocab_file (str): filename of the vocabulary text file
            vocab_dir (str): path to the vocabulary file 
        Returns:
            str: the decoded sentence.
        """
        # Remove the dimensions of size 1
        integers = list(np.squeeze(integers))
        # Set the encoding of the "end of sentence" as 1
        EOS = 1
        # Remove the EOS to decode only the original tokens
        if EOS in integers:
            integers = integers[:integers.index(EOS)]  
        return trax.data.detokenize(integers, vocab_file=vocab_file, vocab_dir=vocab_dir)

    def next_symbol(model, input_tokens, cur_output_tokens, temperature):
        """Returns the index of the next token.
        Args:
            model: the NMT model.
            input_tokens (np.ndarray 1 x n_tokens): tokenized representation of the input sentence
            cur_output_tokens (list): tokenized representation of previously translated words
            temperature (float): parameter for sampling ranging from 0.0 to 1.0.
                0.0: same as argmax, always pick the most probable token
                1.0: sampling from the distribution (can sometimes say random things)
        Returns:
            int: index of the next token in the translated sentence
            float: log probability of the next symbol
        """
        # set the length of the current output tokens
        token_length = len(cur_output_tokens)
        # calculate next power of 2 for padding length 
        padded_length = np.power(2, int(np.ceil(np.log2(token_length + 1))))
        # pad cur_output_tokens up to the padded_length
        padded = cur_output_tokens + [0] * (padded_length - token_length) 
        # model expects the output to have an axis for the batch size in front so
        # convert `padded` list to a numpy array with shape (x, <padded_length>) where the
        # x position is the batch axis.
        padded_with_batch = np.expand_dims(padded, axis=0)
        # the model prediction.
        output, _ = model((input_tokens, padded_with_batch))   
        # get log probabilities from the last token output
        log_probs = output[0, token_length, :]
        # get the next symbol by getting a logsoftmax sample
        symbol = int(tl.logsoftmax_sample(log_probs, temperature))
        return symbol, float(log_probs[symbol])

    def sampling_decode(input_sentence, model = None, temperature=0.0, vocab_file=None, vocab_dir=None):
        """Returns the translated sentence.
        Args:
            input_sentence (str): sentence to translate.
            model: the NMT model.
            temperature (float): parameter for sampling ranging from 0.0 to 1.0.
                0.0: same as argmax, always pick the most probable token
                1.0: sampling from the distribution (can sometimes say random things)
            vocab_file (str): filename of the vocabulary
            vocab_dir (str): path to the vocabulary file
        Returns:
            tuple: (list, str, float)
                list of int: tokenized version of the translated sentence
                float: log probability of the translated sentence
                str: the translated sentence
        """     
        # encode the input sentence
        input_tokens = tokenize(input_sentence, vocab_file=vocab_file, vocab_dir=vocab_dir)
        # initialize the list of output tokens
        cur_output_tokens = []
        # initialize an integer that represents the current output index
        cur_output = 0  
        # Set the encoding of the "end of sentence" as 1
        EOS = 1
        # check that the current output is not the end of sentence token
        while cur_output != EOS: 
            # update the current output token by getting the index of the next word
            cur_output, log_prob = next_symbol(model, input_tokens, cur_output_tokens, temperature)
            # append the current output token to the list of output tokens
            cur_output_tokens.append(cur_output) 
            #detokenize(cur_output, vocab_file=vocab_file, vocab_dir=vocab_dir)
        # detokenize the output tokens
        sentence = detokenize(cur_output_tokens, vocab_file=vocab_file, vocab_dir=vocab_dir)
        return cur_output_tokens, log_prob, sentence

    def greedy_decode_test(sentence, model=model, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR):
        """Prints the input and output of our NMT model using greedy decode
        Args:
            sentence (str): a custom string.
            model: the NMT model.
            vocab_file (str): filename of the vocabulary
            vocab_dir (str): path to the vocabulary file
        Returns:
            str: the translated sentence
        """    
        _,_, translated_sentence = sampling_decode(sentence, model, vocab_file=vocab_file, vocab_dir=vocab_dir)   
        #print("English: ", sentence)
        #print("German: ", translated_sentence)
        return translated_sentence

    def generate_samples(sentence, n_samples, model=None, temperature=0.6, vocab_file=None, vocab_dir=None):
        """Generates samples using sampling_decode()
        Args:
            sentence (str): sentence to translate.
            n_samples (int): number of samples to generate
            model: the NMT model.
            temperature (float): parameter for sampling ranging from 0.0 to 1.0.
                0.0: same as argmax, always pick the most probable token
                1.0: sampling from the distribution (can sometimes say random things)
            vocab_file (str): filename of the vocabulary
            vocab_dir (str): path to the vocabulary file      
        Returns:
            tuple: (list, list)
                list of lists: token list per sample
                list of floats: log probability per sample
        """
        # define lists to contain samples and probabilities
        samples, log_probs = [], []
        # run a for loop to generate n samples
        for _ in range(n_samples):
            # get a sample using the sampling_decode() function
            sample, logp, _ = sampling_decode(sentence, model, temperature, vocab_file=vocab_file, vocab_dir=vocab_dir)
            # append the token list to the samples list
            samples.append(sample)
            # append the log probability to the log_probs list
            log_probs.append(logp)               
        return samples, log_probs


    def jaccard_similarity(candidate, reference):
        """Returns the Jaccard similarity between two token lists
        Args:
            candidate (list of int): tokenized version of the candidate translation
            reference (list of int): tokenized version of the reference translation
        Returns:
            float: overlap between the two token lists
        """  
        # convert the lists to a set to get the unique tokens
        can_unigram_set, ref_unigram_set = set(candidate), set(reference)  
        # get the set of tokens common to both candidate and reference
        joint_elems = can_unigram_set.intersection(ref_unigram_set)
        # get the set of all tokens found in either candidate or reference
        all_elems = can_unigram_set.union(ref_unigram_set)
        # divide the number of joint elements by the number of all elements
        overlap = len(joint_elems) / len(all_elems)
        return overlap


    def rouge1_similarity(system, reference):
        """Returns the ROUGE-1 score between two token lists
        Args:
            system (list of int): tokenized version of the system translation
            reference (list of int): tokenized version of the reference translation
        Returns:
            float: overlap between the two token lists
        """    
        # make a frequency table of the system tokens
        sys_counter = Counter(system)   
        # make a frequency table of the reference tokens
        ref_counter = Counter(reference)
        # initialize overlap to 0
        overlap = 0
        # run a for loop over the sys_counter object
        for token in sys_counter:      
            # lookup the value of the token in the sys_counter dictionary 
            token_count_sys = sys_counter.get(token,0)
            # lookup the value of the token in the ref_counter dictionary 
            token_count_ref = ref_counter.get(token,0)
            # update the overlap by getting the smaller number between the two token counts above
            overlap += min(token_count_sys, token_count_ref) 
        # get the precision (i.e. number of overlapping tokens / number of system tokens)
        precision = overlap / sum(sys_counter.values())    
        # get the recall (i.e. number of overlapping tokens / number of reference tokens)
        recall = overlap / sum(ref_counter.values()) 
        if precision + recall != 0:
            # compute the f1-score
            rouge1_score = 2 * ((precision * recall)/(precision + recall))
        else:
            rouge1_score = 0 
        return rouge1_score

    def average_overlap(similarity_fn, samples, *ignore_params):
        """Returns the arithmetic mean of each candidate sentence in the samples
        Args:
            similarity_fn (function): similarity function used to compute the overlap
            samples (list of lists): tokenized version of the translated sentences
            *ignore_params: additional parameters will be ignored
        Returns:
            dict: scores of each sample
                key: index of the sample
                value: score of the sample
        """    
        # initialize dictionary
        scores = {}
        # run a for loop for each sample
        for index_candidate, candidate in enumerate(samples):    
            # initialize overlap to 0.0
            overlap = 0.0
            # run a for loop for each sample
            for index_sample, sample in enumerate(samples): 
                # skip if the candidate index is the same as the sample index
                if index_candidate == index_sample:
                    continue                
                # get the overlap between candidate and sample using the similarity function
                sample_overlap = similarity_fn(candidate,sample)            
                # add the sample overlap to the total overlap
                overlap += sample_overlap            
            # get the score for the candidate by computing the average
            score = overlap/index_sample        
            # save the score in the dictionary. use index as the key.
            scores[index_candidate] = score        
        return scores


    def weighted_avg_overlap(similarity_fn, samples, log_probs):
        """Returns the weighted mean of each candidate sentence in the samples
        Args:
            samples (list of lists): tokenized version of the translated sentences
            log_probs (list of float): log probability of the translated sentences
        Returns:
            dict: scores of each sample
                key: index of the sample
                value: score of the sample
        """
        # initialize dictionary
        scores = {}   
        # run a for loop for each sample
        for index_candidate, candidate in enumerate(samples):          
            # initialize overlap and weighted sum
            overlap, weight_sum = 0.0, 0.0 
            # run a for loop for each sample
            for index_sample, (sample, logp) in enumerate(zip(samples, log_probs)):
                # skip if the candidate index is the same as the sample index            
                if index_candidate == index_sample:
                    continue            
                # convert log probability to linear scale
                sample_p = float(np.exp(logp))
                # update the weighted sum
                weight_sum += sample_p
                # get the unigram overlap between candidate and sample
                sample_overlap = similarity_fn(candidate, sample)           
                # update the overlap
                overlap += sample_p * sample_overlap        
            # get the score for the candidate
            score = overlap / weight_sum
            # save the score in the dictionary. use index as the key.
            scores[index_candidate] = score
        return scores


    def mbr_decode(sentence, n_samples=4, score_fn=weighted_avg_overlap, similarity_fn=rouge1_similarity, model=model,
                temperature=0.6, vocab_file=VOCAB_FILE, vocab_dir=VOCAB_DIR):
        """Returns the translated sentence using Minimum Bayes Risk decoding
        Args:
            sentence (str): sentence to translate.
            n_samples (int): number of samples to generate
            score_fn (function): function that generates the score for each sample
            similarity_fn (function): function used to compute the overlap between a
            pair of samples
            model: the NMT model.
            temperature (float): parameter for sampling ranging from 0.0 to 1.0.
                0.0: same as argmax, always pick the most probable token
                1.0: sampling from the distribution (can sometimes say random things)
            vocab_file (str): filename of the vocabulary
            vocab_dir (str): path to the vocabulary file
        Returns:
            str: the translated sentence
        """
        # generate samples
        samples, log_probs = generate_samples(sentence, n_samples,
                                            model, temperature,
                                            vocab_file, vocab_dir)   
        # use the scoring function to get a dictionary of scores
        scores = score_fn(similarity_fn, samples, log_probs)
        # find the key with the highest score
        max_index = max(scores, key=scores.get) 
        # detokenize the token list associated with the max_index
        translated_sentence = detokenize(samples[max_index], vocab_file, vocab_dir)
        return (translated_sentence, max_index, scores)

    def translator(text):
        output=mbr_decode(text)[0]
        return output
    
    if speed=='Fast':
      #print(text)
      return greedy_decode_test(text)
    else:
      return translator(text)


def Summarize(input,model=model_sum):
    text=input

    def next_symbol(cur_output_tokens, model):
        """Returns the next symbol for a given sentence.

        Args:
            cur_output_tokens (list): tokenized sentence with EOS and PAD tokens at the end.
            model (trax.layers.combinators.Serial): The transformer model.

        Returns:
            int: tokenized symbol.
        """
        ### START CODE HERE (REPLACE INSTANCES OF 'None' WITH YOUR CODE) ###
        
        # current output tokens length
        token_length = len(cur_output_tokens)
        # calculate the minimum power of 2 big enough to store token_length
        # HINT: use np.ceil() and np.log2()
        # add 1 to token_length so np.log2() doesn't receive 0 when token_length is 0
        padded_length = 2**int(np.ceil(np.log2(token_length + 1)))

        # Fill cur_output_tokens with 0's until it reaches padded_length
        padded = cur_output_tokens + [0] * (padded_length - token_length)
        padded_with_batch = np.array(padded)[None, :] # Don't replace this None! This is a way of setting the batch dim

        # model expects a tuple containing two padded tensors (with batch)
        output, _ = model((padded_with_batch, padded_with_batch)) 
        # HINT: output has shape (1, padded_length, vocab_size)
        # To get log_probs you need to index output wih 0 in the first dim
        # token_length in the second dim and all of the entries for the last dim.
        log_probs = output[0, token_length, :]
        
        ### END CODE HERE ###
        
        return int(np.argmax(log_probs))

    def tokenize(input_str, EOS=1):
        """Input str to features dict, ready for inference"""

        # Use the trax.data.tokenize method. It takes streams and returns streams,
        # we get around it by making a 1-element stream with `iter`.
        inputs =  next(trax.data.tokenize(iter([input_str]),
                                            vocab_dir='vocab_dir',
                                            vocab_file='summarize32k.subword.subwords'))

        # Mark the end of the sentence with EOS
        return list(inputs) + [EOS]

    def detokenize(integers):
        """List of ints to str"""
    
        s = trax.data.detokenize(integers,
                                vocab_dir='vocab_dir',
                                vocab_file='summarize32k.subword.subwords')
        
        return wrapper.fill(s)

    def greedy_decode(input_sentence, model=model, next_symbol=next_symbol, tokenize=tokenize, detokenize=detokenize):
        """Greedy decode function.

        Args:
            input_sentence (string): a sentence or article.
            model (trax.layers.combinators.Serial): Transformer model.

        Returns:
            string: summary of the input.
        """
        
        ### START CODE HERE (REPLACE INSTANCES OF 'None' WITH YOUR CODE) ###
        # Use tokenize()
        cur_output_tokens = tokenize(input_sentence) + [0]    
        generated_output = [] 
        cur_output = 0 
        EOS = 1 
        
        while cur_output != EOS:
            # Get next symbol
            cur_output = next_symbol(cur_output_tokens, model)
            # Append next symbol to original sentence
            cur_output_tokens.append(cur_output)
            # Append next symbol to generated sentence
            generated_output.append(cur_output)
            
            #print(detokenize(generated_output))
        
        ### END CODE HERE ###
            
        return detokenize(generated_output)

    #test_sentence = "It was a sunny day when I went to the market to buy some flowers. But I only found roses, not tulips."



    def summarizer(text):
        output=greedy_decode(text,model)
        output=output[:-5] 
        return output
    return summarizer(text)

VOCAB_FILE_FT = 'endefr_32k.subword'
VOCAB_DIR_FT = 'gs://trax-ml/vocabs/'

VOCAB_FILE_GT = 'ende_32k.subword'
VOCAB_DIR_GT = 'gs://trax-ml/vocabs/'

def main_fun(text,option,language=None,speed=None):
  if len(option)==0:
    return 'Add Translate or Summarize as an option'
  if language=='French':
    model=model_ft
    vocab_file=VOCAB_FILE_FT
    vocab_dir=VOCAB_DIR_FT
  elif language=='German':
    model=model_gt
    vocab_file=VOCAB_FILE_GT
    vocab_dir=VOCAB_DIR_GT
  if 'Summarize' in option:
    text=Summarize(text)
  if 'Translate' in option:
    text=Translate(text,model=model,VOCAB_FILE=vocab_file,VOCAB_DIR=vocab_dir,speed=speed)

  return text
test='It was a sunny day when I went to the market to buy some flowers. But I only found roses, not tulips.'
print(main_fun(test,option=['Translate','Summarize'],language='French',speed='Fast'))

title = "News Article Translator and Summarizer"
description = "Transformer language model for summarizing and translating news clippings into various languages. To use the model, enter a news clipping into the text box and select the language for translating or summarizing. Checkout the examples and usage instructions here."

demo = gr.Interface(
    main_fun,
    inputs=
    [
        gr.Textbox(
            label="Input",
            lines=6,
            value="It was a sunny day when I went to the market to buy some flowers. But I only found roses, not tulips.",
        ),
        gr.CheckboxGroup(["Translate", "Summarize"]),
        gr.Dropdown(["French", "German"],value='French'),
        gr.Radio(['Fast','Slow'], type="value", value='Fast', label='Speed')
    ],
    outputs=[ gr.Textbox(
            label="Output",
            lines=6,
        )],
    title=title,
    description=description)
demo.launch()