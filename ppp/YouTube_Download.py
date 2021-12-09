#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes : NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7 : Thu Mar  12 10:40:23 2020
"""

def youtube_download(link, file_dir='', wd=''):
    from pytube import YouTube
    import os

    YouTube(link).streams.first().download(os.path.join(wd, file_dir))

if __name__ == '__main__':
    link = 'https://www.youtube.com/watch?v=7c9edu0LAHc&ab_channel=LauraH' #'https://www.youtube.com/watch?v=qna0Wy-Xf8I&t=312s'
    youtube_download(link)
