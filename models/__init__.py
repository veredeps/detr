# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .redetectr import build as build_redetectr


def build_model(args):
    return build(args)

def build_redetectr_model(args):
    return build_redetectr(args)