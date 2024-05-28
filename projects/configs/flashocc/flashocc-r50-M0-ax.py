_base_ = ['./flashocc-r50-M0.py',
          ]

model = dict(
    type='BEVDetOCCAX',
    wocc=True,
    wdet3d=False,
)
