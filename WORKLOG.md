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
