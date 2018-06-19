""" evaluation scripts"""
import logging
import os
import re
import subprocess as sp
from os.path import join
from os.path import normpath, basename

from cytoolz import curry
from pyrouge import Rouge155
from pyrouge.utils import log

try:
    _ROUGE_PATH = os.environ['ROUGE']
except KeyError:
    print('Warning: ROUGE is not configured')
    _ROUGE_PATH = None


def eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir,
               cmd='-c 95 -r 1000 -n 2 -m', system_id=1, force=False):
    """ evaluate by original Perl implementation"""
    # silence pyrouge logging
    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    rouge_dec = join(dec_dir, '../rouge_dec')
    if not os.path.exists(rouge_dec) or force:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, rouge_dec)
    rouge_ref = join(ref_dir, '../rouge_{}_ref'.format(basename(normpath(ref_dir))))
    if not os.path.exists(rouge_ref) or force:
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, rouge_ref)
    rouge_settings = join(dec_dir, '../rouge_settings.xml')
    if not os.path.exists(rouge_settings) or force:
        Rouge155.write_config_static(
            rouge_dec, dec_pattern,
            rouge_ref, ref_pattern,
            rouge_settings, system_id
        )
        cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
               + cmd
               + ' -a {}'.format(rouge_settings))
    output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output


try:
    _METEOR_PATH = os.environ['METEOR']
except KeyError:
    print('Warning: METEOR is not configured')
    _METEOR_PATH = None


def eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir, force=False):
    """ METEOR evaluation"""
    assert _METEOR_PATH is not None
    ref_matcher = re.compile(ref_pattern)
    refs = sorted([r for r in os.listdir(ref_dir) if ref_matcher.match(r)],
                  key=lambda name: int(name.split('.')[0]))
    dec_matcher = re.compile(dec_pattern)
    decs = sorted([d for d in os.listdir(dec_dir) if dec_matcher.match(d)],
                  key=lambda name: int(name.split('.')[0]))

    @curry
    def read_file(file_dir, file_name):
        with open(join(file_dir, file_name)) as f:
            return ' '.join(f.read().split())

    meteor_dec = join(dec_dir, '../meteor_dec.txt')
    if not os.path.exists(meteor_dec) or force:
        with open(meteor_dec, 'w') as dec_f:
            dec_f.write('\n'.join(map(read_file(dec_dir), decs)) + '\n')

    meteor_ref = join(ref_dir, '../meteor_{}_ref.txt'.format(basename(normpath(ref_dir))))
    if not os.path.exists(meteor_ref) or force:
        with open(meteor_ref, 'w') as ref_f:
            ref_f.write('\n'.join(map(read_file(ref_dir), refs)) + '\n')

    cmd = 'java -Xmx2G -jar {} {} {} -l en -norm'.format(
        _METEOR_PATH, meteor_dec, meteor_ref)
    output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output
