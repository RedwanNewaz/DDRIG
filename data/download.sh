#!/usr/bin/zsh

download()
{
 scp -r aredwann@gulfscei-linux.cs.uno.edu:/home/aredwann/PyDev/DDRIG/results/$1 .
}

download $1