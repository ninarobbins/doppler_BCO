#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import bz2
import gzip
from contextlib import contextmanager
import struct
import numpy as np
import dask 
import dask.array as da

import xarray as xr

@contextmanager
def open_zspc(filename):
    with bz2.open(filename, "rb") as gzfile:
        with gzip.open(gzfile, "rb") as infile:
            yield infile


# %%


def read_zspc(filename):
    with open_zspc(filename) as infile:
        header = infile.read(1024)
        global_attrs = dict(zip(
            ["name", "time", "oper", "place", "desc"],
            [v.strip(b'\0').decode("utf-8").strip()
             for v in struct.unpack("<32s32s64s128s256s512x", header)]))

        mainsig = None

        while True:
            #print(infile.tell())
            sig, size = struct.unpack("<4sI", infile.read(8))
            #print(sig,size)
            if mainsig is None or sig == mainsig:
                mainsig = sig
                # main chunk
                pass
            else:
                # sub chunk
                infile.seek(size, 1)

        return xr.Dataset({},
                          attrs=global_attrs)


# %%


def analyze_block_structure(data):
    mainsig = data[1024:1028]

    mainblocks = []
    block_start = 1032
    
    good_sub_signatures = {
        b"PPAR",
        b"SRVI",
        b"HSDV",
        b"ZSPY",
        b"COFA",
    }
    while block_start > 0:
        block_end = data.find(mainsig, block_start)
        while (block_end > 0) and (data[block_end+8:block_end+12] not in good_sub_signatures):    
            block_end = data.find(mainsig, block_end + 1)         
        if block_end > 0:
            next_start = block_end + 8
        else:
            block_end = len(data)
            next_start = -1

        mainblocks.append({
            "tag": mainsig.decode("ascii"),
            "start": block_start,
            "end": block_end,
            "reported_size": struct.unpack("<I", data[block_start-4:block_start])[0],
            "blocks": [],
        })
        block_start = next_start

    for block in mainblocks:
        ofs = block["start"]
        while ofs < block["end"]:
            sig, size = struct.unpack("<4sI", data[ofs:ofs+8])
            end = min(ofs + 8 + size, block["end"])
            #print(sig)
            block["blocks"].append({
                "tag": sig.decode("ascii"),
                "start": ofs + 8,
                "end": end,
                "reported_size": size,
            })
            ofs += size + 8

    return mainblocks


# %%


def decode_ppar(ppar_data):
    return dict(zip(
        ["prf", "pdr", "sft", "avc", "ihp","chg","pol","att","tx",
         "ADCgain0","ADCgain1","wnd","pos","add","len","cal","nos",
        "of0","of1","swt","sum","osc","tst","cor","ofs","HSn","HSa",
        "CalibrPower_M","CalibrSNR_M","CalibrPower_S","CalibrSNR_S",
        "Raw_Gate1","Raw_Gate2","Raw","Prc"],
        struct.unpack("<9i2f15i5f4i", ppar_data)))


# %%


def decode_srvi(srvi_data):
    srvi = dict(zip(["frm", "Tm", "TPow", "NPw1", "NPw2", "CPw1", "CPw2", "PS_Stat",
                     "RC_Err", "TR_Err", "dwSTAT", "dwGRST", "AzmPos", "AzmVel", "ElvPos",
                     "ElvVel", "NorthAngle", "time_milli", "PD_DataQuality", "LO_Frequency", "DetuneFine"],
                    struct.unpack("<2I5f5I5f2L2f", srvi_data)))
    srvi["time"]= np.datetime64("1970-01-01")+np.timedelta64(srvi["Tm"],"s")+np.timedelta64(srvi["time_milli"],"us")
    return srvi


# %%


def decode_hsdv(hsdv_data):
    return np.frombuffer(hsdv_data,"<f4").reshape(2,-1)


# %%


def decode_cofa(cofa_data):
    return np.frombuffer(cofa_data,"<f4").reshape(2,-1)


# %%


@dask.delayed  # decorator 
def decode_zspy(zspy_data,n_gates,n_fft,n_channel=2):
    ofs = 0
    power = np.full((n_channel,n_gates,n_fft),np.nan,dtype="f4")
    for i_gate in range(n_gates):
        n_pieces, = struct.unpack("<h",zspy_data[ofs:ofs+2])
        ofs += 2
        #print(n_pieces)
        for i_piece in range(n_pieces):
            bin_index, n_bins = struct.unpack("<hh",zspy_data[ofs:ofs+4])
            ofs += 4
            for i_channel in range(n_channel):
                piece = np.frombuffer(zspy_data,"<u2",n_bins,ofs)
                ofs += n_bins*2
                piece_max, = struct.unpack("<f",zspy_data[ofs:ofs+4])
                ofs += 4
                #print(piece_max,piece)
                power[i_channel,i_gate,bin_index:bin_index+n_bins] = piece_max*(piece/65530.)
            ofs += 2*((n_bins*2)+4) # dropping real and imaginary part of COCX in ZSPY data structure
    return power


# %%


def read_zspc_using_search(filename):
    with open_zspc(filename) as infile:
        data = infile.read()

    header = data[:1024]
    global_attrs = dict(zip(
        ["name", "time", "oper", "place", "desc"],
        [v.strip(b'\0').decode("utf-8").strip()
         for v in struct.unpack("<32s32s64s128s256s512x", header)]))

    blocks = analyze_block_structure(data)

    def report_block(b, space):
        size = b["end"] - b["start"]
        if size != b["reported_size"]:
            delta = b["reported_size"] - size
            w = f" !=  {b['reported_size']} !!! diff: {delta}"
        else:
            w = ""
        #print(f"{space}{b['tag']}: {b['start']} ... {b['end']}   {size}{w}")
    ppar = None
    srvi = []
    hsdv = []
    cofa = []
    zspy = []
    for mb in blocks:
        #report_block(mb, "")
        for b in mb['blocks']:
            #report_block(b, "    ")
            if b["tag"] == "PPAR":
                ppar_new = decode_ppar(data[b["start"]:b["end"]])
                if ppar is not None and ppar_new != ppar:
                    #pass
                    raise ValueError("found different PPAR element")
                ppar = ppar_new
            if b["tag"] == "SRVI":
                srvi.append(decode_srvi(data[b["start"]:b["end"]]))
            if b["tag"] == "HSDV":
                hsdv.append(decode_hsdv(data[b["start"]:b["end"]]))
            if b["tag"] == "COFA":
                cofa.append(decode_cofa(data[b["start"]:b["end"]]))    
            if b["tag"] == "ZSPY":
                zspy.append(da.from_delayed(
                    decode_zspy(data[b["start"]:b["end"]],ppar["chg"]-2,ppar["sft"]),
                    dtype="f4",
                    shape=(2,ppar["chg"]-2,ppar["sft"])))
    return xr.Dataset({
        "time":(("time",),[s["time"] for s in srvi]),
        "hsdv":(("time","channel","range"),np.stack(hsdv,axis=0)),
        "cofa":(("time","channel","range"),np.stack(cofa,axis=0)),
        "power":(("time","channel","range","fftline"),da.stack(zspy,axis=0)), # dask stack !!
        "RadarConst5":(("time",),[t["DetuneFine"] for t in srvi]),
        "NPW1":(("time",),[t["NPw1"] for t in srvi])
       
        },
        attrs={**global_attrs,**ppar})
