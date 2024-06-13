_base_ = ['./flashocc-r50-M0.py',
          ]

model = dict(
    type='BEVDetOCCAXMAXN',
    wocc=True,
    wdet3d=False,
)
