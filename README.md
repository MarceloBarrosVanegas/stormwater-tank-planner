# swmm-tank-optimizer

A research-oriented toolbox for planning urban **stormwater tanks** using
cost–benefit optimization. The workflow connects a 1D SWMM model, GIS
layers with land and construction costs, and evolutionary algorithms
to identify where and how large to build stormwater tanks that control
overflow and reduce flood and environmental damages.

## What this repo does

- Identifies **critical nodes with overflow** in a SWMM drainage network.
- Links each critical node to **candidate tank sites** based on GIS polygons
  (land availability, cost per m², construction difficulty, etc.).
- Represents each candidate as a combination of:
  - overflow node(s),
  - potential tank location,
  - required **diversion** (length of new conveyance).
- Estimates **investment costs**, including:
  - diversion works,
  - new sanitary sewers (when existing pipes are converted to storm sewers),
  - stormwater tank construction,
  - land acquisition/use,
  - restoration of affected infrastructure,
  - (optionally) operation and maintenance.
- Quantifies **benefits** in terms of:
  - reduced direct and indirect **flood damages**,
  - avoided **environmental and morphological impacts** on the receiving river,
  - extended **service life of existing drainage infrastructure**
    (avoided or deferred upsizing due to population growth, increased
    imperviousness and more intense rainfall).
- Runs an **evolutionary optimization** (e.g. differential evolution or NSGA-II)
  to:
  - minimize total investment cost **subject to** a target reduction in
    expected flood/environmental damage, or
  - maximize net benefits (benefits – costs).

The final output is a set of recommended **stormwater tank configurations**
(locations and volumes) that provide a transparent trade-off between
investment costs and risk reduction in an urban catchment.
