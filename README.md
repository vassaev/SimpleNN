## About

Simple Neuron Network library provides an implementation of the Neuron Network.

This library is extremely short and easy to use.

## Resources

- [Simple Neuron Network library](https://github.com/vassaev/SimpleNN)

## Quick Start Guide

Add NLayer.py and NNet.py in your python's project and use like in Example.py:
1. Define an activation and a derivative of the activation functions
2. Create your Neuron Network like this: 
n = NNet(<size of input vector>, <list of sizes of inner layers>, <size of output vector>, <init function for vector>, <init function for matrix>)
3. Train you Neuron Network using prepared data - use 
n.train(<array of vectors of input data>, <array of vectors of output data>, <activation function>, <derivative of the activation function>, <training speed>, <count of iterations>)
4. Then you can use "propagation" function of your Neuron Network to find new solutions

## Reporting Issues

I would be grateful for feedback. If you would like to report a bug, suggest an enhancement or ask a question then please [create a new issue](https://github.com/vassaev/SimpleNN/issues/new).