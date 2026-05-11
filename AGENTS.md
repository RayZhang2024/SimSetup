# SimSetup Agent Notes

- Keep diffraction analysis code modular under `diffraction/`; avoid adding fitting or file-format logic directly to `gui_app.py`.
- Do not copy Mantid, GSAS-II, or Open GENIE source into this repository. Optional external integrations should call the user's local installation.
- Treat files under `Hist/` as reduced focused histogram examples. Use `Hist/ENGINX373922_1.his` for Ni fitting validation.
- Keep `gui_app.py` integration thin: import the diffraction tab and add it to the main tab widget.
