video_zoom4x()
{
    python demo/restoration_video_demo.py \
        configs/basicvsr_plusplus_reds4.py \
        chkpts/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth \
        data/girl \
        results/zoom4x_demo_000
}

video_deblur()
{
    python demo/restoration_video_demo.py \
        configs/basicvsr_plusplus_deblur_dvd.py \
        chkpts/basicvsr_plusplus_deblur_dvd-ecd08b7f.pth \
        data/girl \
        --max-seq-len 12 \
        results/deblur_demo_000
}

video_denoise()
{
    python demo/restoration_video_demo.py \
        configs/basicvsr_plusplus_denoise.py \
        chkpts/basicvsr_plusplus_denoise-28f6920c.pth \
        data/girl \
        --max-seq-len 12 \
        results/denoise_demo_000
}

# video_zoom4x
# video_deblur
video_denoise
