# CO2 emission data

1. Default emission in GEOS-Chem **(Global, 1x1 monthly)**
- Biomass burning: GFED, 1997-2018, 0.25, 6 vegetation types
- **Fossil fuel: CDIAC 1x1 national, 1980-2014**
- **MIX for Asia, 4 categories: industry, power, residential and transport**
- Ocean exchange: climatological exchange for 2000 [Takashi et al., 2009]
- Balance biosphere exchange: CASA, SiB3 **→ ORCHIDEE**
- Net terrestrial exchange: TransCom annual net/residual terrestrial biospheric CO2 [Baker et al., 2006]
- Ship emissions: ICOADS, 2004 **→ EDGAR**
- Aviation emissions: AEIC, aviation fuel burn spatial and seasonal distribution [Simone et al. 2013, Olesen et al., 2013] **→ EDGAR**
- CO2 surface correction for CO oxidation: to subtract from emissions [Nassar et al. 2010] **→ turn off**

2. EDGAR 4.3.2 high resolution (Global, anthropogenic, sector-specific, 2015, 0.1x0.1)

- Source: CHE reports, downloaded
- Unit: ton CO2/yr/gridcell for each sector
- Sectors: AGS (agricultural soils), CHE (chemical processes), ENE (energy generation), FFF (fossil fuel fires), IND (industrial manufacturing), IRO (iron & steel processes), NEU (non-energy use), NFE (non-ferrous metals production), NMM (non-metallic mineral processes), PRO (fossil fuel production), PRU-SOL (products use and solvents), RCO (energy for buildings), REF-TRF (refineries and transformation), SWD-INC (waste incineration), TNR_Aviation_LTO/CDS/CRS/LTO (aviation at 3 height levels: landing-takeoff / climbing-descent / cruise), TNR-Other (non-road transport over land), TNR_ship (shipping), TRO (road transport).)

3. EDGAR 5 high resolution (Global, anthropogenic, sector-specific, until **2018, 0.1x0.1**)

- Source: EDGAR website, downloaded

[Fossil CO2 & GHG emissions of all world countries, 2017](https://edgar.jrc.ec.europa.eu/overview.php?v=CO2andGHG1970-2016)

4. TNO GHGco emission inventory

- EU regional, annual, fossil fuel and biofuel CO2, 1km x 1km (2015), 6km x 6km (2005-2018)
- Source: TNO
- Categories: CO2_ff, CO2_bf, CH4, CO_ff, CO_bf, NOx, NMVOC
- 1-dimensional NetCDF file
~
~
