# Repository release checklist

Complete these items before announcing the repository as reusable research software.

- [ ] Replace `ADD_GITHUB_RELEASE_URL` in `README.md` with the actual GitHub Release URL for the full evidence archive.
- [ ] Confirm the paper PDF is the final approved version and the repository paper path is correct.
- [ ] Confirm the Kaggle model dataset is public and contains the exact two checkpoints and canonical split manifest expected by the notebook.
- [ ] Run the notebook from a clean Kaggle session with only the public model-artifact dataset and the separately obtained original source dataset attached.
- [ ] Verify the checkpoint and manifest hashes reported by the notebook.
- [ ] Check the Charisis dataset licence for conditions on distributing derived model weights. Do not assume that permission to use the dataset automatically permits public redistribution of trained weights.
- [ ] Choose and add a code licence. No code licence is included in this starter package because the project maintainers have not selected one.
- [ ] Confirm the licence/redistribution terms for both model checkpoints. These may differ from the code licence.
- [ ] If the model weights cannot be redistributed, change the Kaggle dataset visibility/access or obtain permission before public release.
- [ ] Add a versioned GitHub Release and, if desired, enable GitHub-Zenodo archiving to obtain a DOI. Add the DOI to `CITATION.cff` only after it exists.
- [ ] Review author names and preferred citation metadata before publishing.

## Licensing note

A public GitHub repository without a licence is not an open-source licence grant. In general, copyright remains with the rights holder and others do not receive broad permission to reuse, modify, or redistribute the code. Do not add an MIT, Apache-2.0, or other licence merely as a placeholder; choose one deliberately and confirm that all contributors agree.
