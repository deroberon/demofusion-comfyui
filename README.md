# ComfyUI Demofusion Custom Node

## Introduction

This is my implementation of the work and algorithms described here by Ruoyi Du:
    * https://ruoyidu.github.io/demofusion/demofusion.html
    * https://github.com/PRIS-CV/DemoFusion

My idea was wrapping Demofusion in a node to be used in ComfyUI, so we could use this amazing tool to experimenting with this technique. 

As it it my first ComfyUI custom node, I'm not sure if I'm implementing the best practices, specially with respect to:
    * How to install python missing needed packages

So, suggestions are super welcome. I also would like to implement other features but I was not able to figure out how to do them yet: 
    * To use safetensors local files and not only Hugging Face models, but I was not able yet to integrate the models used in diffusers python library with the ones used in Comfyui nodes.

If anyone have some ideas about how to do it, again, thank you very much for yor collaborating and tips.

## Installing
To install this node, is just like any other one, no special procedures are needed:
    * Git clone the repository in the ComfyUI/custom_nodes folder
    * Restart ComfyUI

It's also good to remember that this technique requires a lot of VRAM (plus 18G). 

## Under development
This node is under development, so use it at your own risk. 
