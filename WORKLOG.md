# 10/31/2026
## Jonathan 
## Documentation for `CBP_imputation/CBP_data`

Created a new local folder, `alder/cbp`, to store all data files and code for constructing the CBP datasets and imputations. Data source: [Census CBP datasets](https://www.census.gov/programs-surveys/cbp/data/datasets.html?text-list-cc4db501b6%3Atab=all#text-list-cc4db501b6).

- Archived CBP datasets from before 1986 are available from the [National Archives](https://catalog.archives.gov/id/613576), in case we're interested.
    - `cbp/cbp_census_pull_api.ipynb` pulls data files using the Census API.
    - `cbp/cbp_census_pull_bulk.ipynb` pulls the CBP county zip files for each year from 1986–2023, hosted at `https://www2.census.gov/programs-surveys/cbp/datasets/<year>/cbp<yy>co.zip`. It doesn't require a Census API key. It creates `cbp/data_bulk`, which stores all the zip files.
    - `cbp/cbp_create_dataset.ipynb` concatenates all the zip files in `cbp/data_bulk` into one dataset. It doesn't drop any rows or columns; it only consolidates the files. The output is `cbp/cbp_county_1986_2023.parquet`, at the county-industry-year level. The notebook's markdown cells cover duplicate rows, column availability, and county code `999`.
    - Variable name guides: [ProximityOne CBP guide](https://proximityone.com/dataresources/guide/index.html?dr_pds_cbp.htm) and [Census CBP glossary](https://www.census.gov/programs-surveys/cbp/about/glossary.html).

# 10/30/2026
## Andrew & Jonathan Meeting
Before Monday: 
- Jonathan:
    - Download and consolidate CBP data into single data set
    - harmonize industry classifications across years (from SIC to NAICS to make adding recent years easier) using https://www.fpeckert.me/cbp/
    - deal with synthetic noise, flags, and ranges in CBP using method 2: Eckert, Fort, Schott, and Yang (available at https://www.fpeckert.me/efsy.pdf)
- Andrew:
    - harmonize industry classifications across years (from SIC to NAICS to make adding recent years easier) using https://ddorn.net/data.htm
    - deal with synthetic noise, flags, and ranges in CBP using method 1: Autor, Dorn, & Hansen: imputation is described in https://ddorn.net/papers/Autor-Dorn-Hanson-ChinaSyndrome-Appendix.pdf     
- Yet to do: County-to-CZ crosswalk files are available at https://ddorn.net/data.htm [section E]


# 10/23/2026
## Andrew & Jonathan
- Download and consolidate CBP data into single data set
- Collapse data to CZ level:
    - harmonize industry classifications across years (from SIC to NAICS to make adding recent years easier): cross-walks going back in time are available at https://ddorn.net/data.htm [Section C] but we need the opposite direction
    - deal with synthetic noise, flags, and ranges in CBP:
        - method 1: Autor, Dorn, & Hansen: imputation is described in https://ddorn.net/papers/Autor-Dorn-Hanson-ChinaSyndrome-Appendix.pdf
        - method 2: Eckert, Fort, Schott, and Yang (available at https://www.fpeckert.me/efsy.pdf)
        - method 3: apply for restricted LBD data (which is used in paper by Bloom et al. available at https://drive.google.com/file/d/1BowvWl5lRuZXV-3JU5Qh_2qUlTbzhorP/view?usp=sharing)
    - County-to-CZ crosswalk files are available at https://ddorn.net/data.htm [section E]
- Jonathan:
    - increase number of sectors and location; keep track of run time
    - calculate HHI at local level (across sectors)
- Andrew: play in the sandbox!
## Simeon
Read the literature on empirical and quantitative approaches to identify the magnitude of local within-industry spillovers in the data.

# 10/14/2026

## Team
Fifth meeting of the fall semester with Jonathan and Andrew. 

#### To-do list for next meeting on Friday, October 24
- Andrew reads through the lecture notes, discusses with Jonathan if finishes before the next meeting. 
- Jonathan experiments with different values of $\psi \in [0,0.25]$ and ways to visually display the results. A first attempt could be to create a heat map of "market concentration" at various levels of $\psi$ and report corresponding HHI values for region-sectors. 


# 10/7/2026

## Team
Fourth meeting of the fall semester with Jonathan and Andrew. 

#### To-do list for next meeting on Wednesday, October 14
- Simeon finishes lecture note on migration shares.
- Andrew hangs tight for now. 
- Jonathan adds the spillover term to the model, and runs some exploratory simulations comparing transition paths, migration, and equilibrium for different values of $\psi$. Experiment with different types of productivity shocks (permanent vs temporary) for different $\psi$, and figure out a nice way to present the results visually.

# 10/2/2026

## Team
Third meeting of the fall semester with Jonathan and Andrew. 

Agglomeration spillover $`(\mathcal{L}^{nj}_t)^\psi`$: Augments the region-sector productivity $`A^{nj}_{t+1}`$ in the intermediate firm's production function

```math
y^{nj}_{t+1}(\omega) = A^{nj}_{t+1}(\omega) \cdot \ell^{nj}_{t+1}(\omega) \cdot \left(\mathcal{L}^{nj}_{t+1}\right)^\psi
```

where $`\omega \in [0,1]`$ denotes the intermediate variety and 
```math
\mathcal{L}^{nj}_{t+1} = \int_{\omega \in \Omega^{nj}_{t+1}} \ell^{nj}_{t+1}(\omega)\, d\omega.
```

#### To-do list for next meeting on Wednesday, October 7
- Simeon finishes lecture note on migration shares.
- Andrew hangs tight for now. Can start playing around with the CDP simplified codes.
- Jonathan gives a first pass at adding the spillover term to the model


# 9/25/2026

## Jonathan
- Uploaded the simplified CDP model Julia files to /cdp-simple on the github. Scripts 01-03 are the ancestor scripts to 04, which implements the dynamic hat algebra into the simplified CDP model.
- Script 05 has been created and it contains the machinery from script 04 and applies it to a set of counterfactual exercises.
- Documentation for script 04,05 has been updated. 

# 9/23/2026

## Team
Second meeting of the fall semester with Jonathan and Andrew.

### Looking forward
- Starting next week, the focus will be on calibrating data to fit the spatial model. Focus will be on shaping the data to the commuting-zone level and imputing values for censored/manipulated data. 

#### To-do list for next meeting on Friday, October 2
- Simeon writes lecture note 5, which combines the migration note with input-output linkages. This note isn't particularly pressing, but is useful in the context of the Caliendo et al. (2019) paper.
- Andrew hangs tight for now. Rereads the Caliendo et al. (2019) paper
- Jonathan cleans up the Julia code:
    + Moves simulated shock/parameter counterfactual scenarios from script 04 to its own sandbox script. 
    + Cleans up the documentation/comments on the relevant codes


# 9/18/2025
## Andrew and Jonathan
Andrew and Jonathan meet to discuss Andrew's progress on the paper and resolve any questions Andrew has.

#### To-do list for next meeting on Wednesday, September 23
- Andrew sits with the dynamic hat algebra for a bit longer, and begins to rewrite the model/equilibrium conditions with our simplifications (no structures, no input-output linkages)
- Jonathan looks over the Julia code to check for accuracy.


# 9/11/2026

## Team
First meeting of the fall semester with Jonathan and Andrew.

#### To-do list for next meeting on Wednesday, September 23
- Simeon writes lecture note 4 covering model with costly migration, sub-national locations (state, CZs,...), but without input output linkages
- Another lecture note (the fifth in the series) combines the migration note with input-output linkages. This note isn't particularly pressing, but is useful in the context of the Caliendo et al. (2019) paper.
- After reading lecture note 4, Andrew reads the Caliendo et al. paper. Jonathan and Andrew then meet to discuss and make sure everyone is on the same page.
- Jonathan is working on the Julia code:
    + check convergence after fundamentals have settled by:
        + varying the number of locations ($N$) and sectors ($J$)
        + varying the migration cost (possibly explore the extreme cases with zero cost and infinite cost, both of which have a steady state with zero net migration flows)
    + in the DataFrame, record gross migration flow at the $(i,j)$ level; check if the net flows converge to zero when growth factors of all endogenous variables converge to 1.
