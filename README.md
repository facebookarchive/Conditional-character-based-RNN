# Conditional character based RNN

This project proposes an efficient character based recurrent neural network. 
The RNN has word level input instead of standard character level input.
The output of thhe RNN is conditioned by the previous words.

## Most important code

There are two models:
* `char-rnn-conditional`
    * This contains the character RNN which has conditional outputs
* `mixed-rnn`
    * This contains the mixed word/char level rnn code

the mixed-rnn code splits up the stream of characters into words using
the character _ . 

## Requirements

This code has been tested on Linux, but should work on any machine. The is no dependencies.

## Join the Conditional character based RNN community

See the [CONTRIBUTING](https://github.com/ritwik12/Conditional-character-based-RNN/blob/master/CONTRIBUTING.md) file for how to help out.

## License
Conditional character based RNN is BSD-licensed. We also provide an additional patent grant.
