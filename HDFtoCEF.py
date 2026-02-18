import pandas as pd 
import polars as pl
#pl.enable_string_cache ## Avoid Polars' CategoricalRemappingWarning
import numpy as np
from collections import OrderedDict
from statistics import median_grouped
print("{} Libraries Imported".format(datetime.now()))
statedict = OrderedDict({'02': 'AK', '01': 'AL', '05': 'AR', '04': 'AZ', '06': 'CA', '08': 'CO', '09': 'CT', '11': 'DC', '10': 'DE', '12': 'FL', '13': 'GA', '15': 'HI', '19': 'IA', '16': 'ID', '17': 'IL', '18': 'IN', '20': 'KS', '21': 'KY', '22': 'LA', '25': 'MA', '24': 'MD', '23': 'ME', '26': 'MI', '27': 'MN', '29': 'MO', '28': 'MS', '30': 'MT', '37': 'NC', '38': 'ND', '31': 'NE', '33': 'NH', '34': 'NJ', '35': 'NM', '32': 'NV', '36': 'NY', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '51': 'VA', '50': 'VT', '53': 'WA', '55': 'WI', '54': 'WV', '56': 'WY', '72':'PR'})
#racedict  = {'01': 'white',  '02': 'black',  '03': 'aian',  '04': 'asian',  '05': 'nhopi',  '06': 'sor',  '07': 'white-black',  '08': 'white-aian',  '09': 'white-asian',  '10': 'white-nhopi',  '11': 'white-sor',  '12': 'black-aian',  '13': 'black-asian',  '14': 'black-nhopi',  '15': 'black-sor',  '16': 'aian-asian',  '17': 'aian-nhopi',  '18': 'aian-sor',  '19': 'asian-nhopi',  '20': 'asian-sor',  '21': 'nhopi-sor',  '22': 'white-black-aian',  '23': 'white-black-asian',  '24': 'white-black-nhopi',  '25': 'white-black-sor',  '26': 'white-aian-asian',  '27': 'white-aian-nhopi',  '28': 'white-aian-sor',  '29': 'white-asian-nhopi',  '30': 'white-asian-sor',  '31': 'white-nhopi-sor',  '32': 'black-aian-asian',  '33': 'black-aian-nhopi',  '34': 'black-aian-sor',  '35': 'black-asian-nhopi',  '36': 'black-asian-sor',  '37': 'black-nhopi-sor',  '38': 'aian-asian-nhopi',  '39': 'aian-asian-sor',  '40': 'aian-nhopi-sor',  '41': 'asian-nhopi-sor',  '42': 'white-black-aian-asian',  '43': 'white-black-aian-nhopi',  '44': 'white-black-aian-sor',  '45': 'white-black-asian-nhopi',  '46': 'white-black-asian-sor',  '47': 'white-black-nhopi-sor',  '48': 'white-aian-asian-nhopi',  '49': 'white-aian-asian-sor',  '50': 'white-aian-nhopi-sor',  '51': 'white-asian-nhopi-sor',  '52': 'black-aian-asian-nhopi',  '53': 'black-aian-asian-sor',  '54': 'black-aian-nhopi-sor',  '55': 'black-asian-nhopi-sor',  '56': 'aian-asian-nhopi-sor',  '57': 'white-black-aian-asian-nhopi',  '58': 'white-black-aian-asian-sor',  '59': 'white-black-aian-nhopi-sor',  '60': 'white-black-asian-nhopi-sor',  '61': 'white-aian-asian-nhopi-sor',  '62': 'black-aian-asian-nhopi-sor',  '63': 'white-black-aian-asian-nhopi-sor'}
racealonedict  = {1: 'White Alone',  2: 'Black Alone',  3: 'AIAN Alone',  4: 'Asian Alone',  5: 'NHOPI Alone',  6: 'Some Other Race Alone',  7: 'Two Or More Races'}
raceincombdict = {'whitealone-or-incomb':'White Alone or In Combination', 'blackalone-or-incomb':'Black Alone or In Combination', 'aianalone-or-incomb':'AIAN Alone or In Combination', 'asianalone-or-incomb':'Asian Alone or In Combination', 'nhopialone-or-incomb': 'NHOPI Alone or In Combination', 'soralone-or-incomb':'Some Other Race Alone or In Combination'}
hispdict = {'1': 'Not Hispanic', '2':'Hispanic'}
sexdict = {'1': 'Male', '2':'Female'}
def make_geolist(df,geo):
    return np.sort(df.select(pl.col(geo)).drop_nulls().unique().to_series().to_numpy())

def make_geodf(geos,geoid):
    return pl.LazyFrame({geoid:pl.Series(geos, dtype=pl.Enum(geos))})

def make_nullrows(df, indexcols):
    newdf=df.select(pl.col(indexcols[0]).unique())
    for curcol in indexcols[1:]:
        newdf=newdf.join(df.select(pl.col(curcol).unique()), how='cross')
    return df.join(newdf, on=indexcols, how='full', coalesce=True)

def calculate_stats(data):
    data = (
        data
        .with_columns(pl.col('HDF_Population').fill_null(0), pl.col('MDF_Population').fill_null(0))
        .with_columns(Diff = pl.col('MDF_Population') - pl.col('HDF_Population'))
        .with_columns(AbsDiff = abs(pl.col('Diff')))
        .with_columns(PercDiff = pl
                      .when((pl.col('HDF_Population') == 0) & (pl.col('MDF_Population') == 0))
                      .then(pl.lit(0))
                      .when(pl.col('HDF_Population') == 0)
                      .then(pl.col('Diff')/((pl.col('HDF_Population') + pl.col('MDF_Population'))/2))
                      .otherwise(pl.col('Diff') / pl.col('HDF_Population'))
                      )
        .with_columns(AbsPercDiff = abs(pl.col('PercDiff')))
    )
    return data

def calculate_ss(data, geography, sizecategory, characteristic):
    if len(data) > 0:
        odf = pd.DataFrame({
            'Geography': geography,
            'Size_Category': sizecategory,
            'Characteristic': characteristic,
            'MinDiff': np.nanmin(data['Diff']),
            'MeanDiff':np.nanmean(data['Diff']),
            'MedianDiff':np.nanmedian(data['Diff']),
            'MaxDiff': np.nanmax(data['Diff']),
            'MeanAbsDiff':np.nanmean(data['AbsDiff']),
            'MedianAbsDiff':np.nanmedian(data['AbsDiff']),
            'AbsDiff90th': np.nanpercentile(data['AbsDiff'], 90),
            'AbsDiff95th': np.nanpercentile(data['AbsDiff'], 95),
            'MinPercDiff': np.nanmin(data['PercDiff']),
            'MeanPercDiff':np.nanmean(data['PercDiff']),
            'MedianPercDiff':np.nanmedian(data['PercDiff']),
            'MaxPercDiff': np.nanmax(data['PercDiff']),
            'PercDiffNAs': data['PercDiff'].is_null().sum(),
            'MeanAbsPercDiff': np.nanmean(data['AbsPercDiff']),
            'MedianAbsPercDiff': np.nanmedian(data['AbsPercDiff']),
            'AbsPercDiff90th': np.nanpercentile(data['AbsPercDiff'], 90),
            'AbsPercDiff95th': np.nanpercentile(data['AbsPercDiff'], 95),
            'AbsPercDiffMax': np.nanmax(data['AbsPercDiff']),
            'AbsPercDiffNAs': data['AbsPercDiff'].is_null().sum(),
            'RMSE': np.sqrt((data['Diff']**2).mean()),
            'CV': 100*(np.sqrt((data['Diff']**2).mean())/np.nanmean(data['HDF_Population'])) if np.nanmean(data['HDF_Population']) != 0 else np.nan,
            'MeanCEFPop': np.nanmean(data['HDF_Population']),
            'NumCells':len(data),
            'NumberBtw2Perc5Perc': len(data.filter((pl.col('AbsPercDiff') >= 2) & (pl.col('AbsPercDiff') <=5))),
            'NumberGreater5Perc':len(data.filter(pl.col('AbsPercDiff') > 5)),
            'NumberGreater200': len(data.filter(pl.col('AbsDiff') > 200)),
            'NumberGreater10Perc':len(data.filter(pl.col('AbsPercDiff') > 10))
        }, index=[0])
    else:
        odf = pd.DataFrame({
            'Geography': geography,
            'Size_Category': sizecategory,
            'Characteristic': characteristic,
            'MinDiff': 0,
            'MeanDiff':0,
            'MedianDiff':0,
            'MaxDiff': 0,
            'MeanAbsDiff':0,
            'MedianAbsDiff':0,
            'AbsDiff90th': 0,
            'AbsDiff95th': 0,
            'MinPercDiff': 0,
            'MeanPercDiff':0,
            'MedianPercDiff':0,
            'MaxPercDiff': 0,
            'PercDiffNAs': 0,
            'MeanAbsPercDiff': 0,
            'MedianAbsPercDiff': 0,
            'AbsPercDiff90th': 0,
            'AbsPercDiff95th': 0,
            'AbsPercDiffMax': 0,
            'AbsPercDiffNAs':0,
            'RMSE': 0,
            'CV': 0,
            'MeanCEFPop':0,
            'NumCells':0,
            'NumberBtw2Perc5Perc':0,
            'NumberGreater5Perc':0,
            'NumberGreater200':0,
            'NumberGreater10Perc':0
        }, index=[0])
    return odf


def dfhdf():
    return pl.scan_ipc('hdf.arrow')

def dfmdf():
    return pl.scan_ipc('mdf.arrow')

# Counties Total Population
hdfcounties_totalpop = dfhdf().group_by('CountyGEOID').agg(HDF_Population = pl.len())
mdfcounties_totalpop = dfmdf().group_by('CountyGEOID').agg(MDF_Population = pl.len())
counties_totalpop =  hdfcounties_totalpop.join(mdfcounties_totalpop, on='CountyGEOID', how='full', coalesce = True).pipe(calculate_stats).collect()
counties_totalpop.to_pandas(use_pyarrow_extension_array=True).to_csv(f"{OUTPUTDIR}/counties_totalpop.csv", index=False)
ss = counties_totalpop.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Total Population")
outputdflist.append(ss)

# Counties Total Population By Size
counties_totalpop = counties_totalpop.with_columns(
    Total_PopSize = pl.col('HDF_Population').cut(breaks = [1000,5000,10_000,50_000,100_000], left_closed = True))
for i in counties_totalpop['Total_PopSize'].cat.get_categories():
    ss = counties_totalpop.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Total Population")
    outputdflist.append(ss)

#
smallestpop = counties_totalpop['Total_PopSize'].cat.get_categories()[0]
print(smallestpop)
counties_lt1000 = counties_totalpop.filter(pl.col('Total_PopSize') == smallestpop).select('CountyGEOID')
print(counties_lt1000)
# counties_lt1000index = pd.Index(counties_lt1000, name="GEOID")

# Do not delete counties_totalpop

# Places Total Population
hdfplaces_totalpop = dfhdf().group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
mdfplaces_totalpop = dfmdf().group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
places_totalpop = allplacesdf.join(hdfplaces_totalpop.join(mdfplaces_totalpop, on='IncPlaceGEOID', how='full',coalesce=True), how='left', on='IncPlaceGEOID').pipe(calculate_stats).collect()
places_totalpop.to_pandas(use_pyarrow_extension_array=True).to_csv(f"{OUTPUTDIR}/places_totalpop.csv", index=False)
ss = places_totalpop.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "Total Population")
outputdflist.append(ss)

# Places Total Population By Size
places_totalpop = places_totalpop.with_columns(Total_PopSize = pl.col('HDF_Population').cut([500,1000,5000,10_000,50_000,100_000,], left_closed=True))
for i in places_totalpop['Total_PopSize'].cat.get_categories():
    ss = places_totalpop.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "Total Population")
    outputdflist.append(ss)

#
smallestpop = places_totalpop['Total_PopSize'].cat.get_categories()[0]
print(smallestpop)
places_lt500 = places_totalpop.filter(pl.col('Total_PopSize') == smallestpop).select('IncPlaceGEOID')
print(places_lt500.height)
# places_lt500index = pd.Index(places_lt500, name="IncPlaceGEOID")

# Tracts Total Population
hdftracts_totalpop = dfhdf().group_by('TractGEOID').agg(HDF_Population = pl.len())
mdftracts_totalpop = dfmdf().group_by('TractGEOID').agg(MDF_Population = pl.len())
tracts_totalpop =  alltractsdf.join(hdftracts_totalpop.join(mdftracts_totalpop, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
tracts_totalpop.to_pandas(use_pyarrow_extension_array=True).to_csv(f"{OUTPUTDIR}/tracts_totalpop.csv", index=False)
ss = tracts_totalpop.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Total Population")
outputdflist.append(ss)

del hdftracts_totalpop
del mdftracts_totalpop
del tracts_totalpop

# Blocks By Rurality
hdfblocks_totalpop = dfhdf().group_by('BlockGEOID').agg(HDF_Population = pl.len())
mdfblocks_totalpop = dfmdf().group_by('BlockGEOID').agg(MDF_Population = pl.len())
blocks_totalpop =  allblocksdf.join(hdfblocks_totalpop.join(mdfblocks_totalpop, on='BlockGEOID', how='full', coalesce=True), on='BlockGEOID',how='left').pipe(calculate_stats).collect()
blocks_totalpop = blocks_totalpop.join(allgeosur, on="BlockGEOID", how="left")
ss = blocks_totalpop.filter(pl.col('UR') == "U").pipe(calculate_ss, geography="Block", sizecategory = "All", characteristic = "Total Population for Urban Blocks")
outputdflist.append(ss)
ss = blocks_totalpop.filter(pl.col('UR') == "R").pipe(calculate_ss, geography="Block", sizecategory = "All", characteristic = "Total Population for Rural Blocks")
outputdflist.append(ss)

del hdfblocks_totalpop
del mdfblocks_totalpop
del blocks_totalpop

# Elem School Districts Total Population
hdfelemschdists_totalpop = dfhdf().group_by(['SchDistEGEOID']).agg(HDF_Population = pl.len())
mdfelemschdists_totalpop = dfmdf().group_by(['SchDistEGEOID']).agg(MDF_Population = pl.len())
elemschdists_totalpop = allelemschdistsdf.join(hdfelemschdists_totalpop.join(mdfelemschdists_totalpop, on='SchDistEGEOID', how='full', coalesce=True), how='left', on='SchDistEGEOID').pipe(calculate_stats).collect()
elemschdists_totalpop.to_pandas().to_csv(f"{OUTPUTDIR}/elemschdists_totalpop.csv", index=False)
ss = elemschdists_totalpop.pipe(calculate_ss, geography="ESD", sizecategory = "All", characteristic = "Total Population")
outputdflist.append(ss)

# Elem School Districts Total Population By Size
elemschdists_totalpop = elemschdists_totalpop.with_columns(Total_PopSize = pl.col('HDF_Population').cut([1000,5000,10000,50_000,100_000], left_closed=True))
for i in elemschdists_totalpop['Total_PopSize'].cat.get_categories():
    ss = elemschdists_totalpop.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="ESD", sizecategory = str(i), characteristic = "Total Population")
    outputdflist.append(ss)

# Sec School Districts Total Population
hdfsecschdists_totalpop = dfhdf().group_by(['SchDistSGEOID']).agg(HDF_Population = pl.len())
mdfsecschdists_totalpop = dfmdf().group_by(['SchDistSGEOID']).agg(MDF_Population = pl.len())
secschdists_totalpop =  allsecschdistsdf.join(hdfsecschdists_totalpop.join(mdfsecschdists_totalpop, on='SchDistSGEOID', how='full', coalesce=True), how='left', on='SchDistSGEOID').pipe(calculate_stats).collect()
secschdists_totalpop.to_pandas().to_csv(f"{OUTPUTDIR}/secschdists_totalpop.csv", index=False)
ss = secschdists_totalpop.pipe(calculate_ss, geography="SSD", sizecategory = "All", characteristic = "Total Population")
outputdflist.append(ss)

# Sec School Districts Total Population By Size
secschdists_totalpop= secschdists_totalpop.with_columns(Total_PopSize = pl.col('HDF_Population').cut([1000,5000,10000,50_000,100_000], left_closed=True))
for i in secschdists_totalpop['Total_PopSize'].cat.get_categories():
    ss = secschdists_totalpop.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="SSD", sizecategory = str(i), characteristic = "Total Population")
    outputdflist.append(ss)

# Uni School Districts Total Population
hdfunischdists_totalpop = dfhdf().group_by(['SchDistUGEOID']).agg(HDF_Population = pl.len())
mdfunischdists_totalpop = dfmdf().group_by(['SchDistUGEOID']).agg(MDF_Population = pl.len())
unischdists_totalpop =  allunischdistsdf.join(hdfunischdists_totalpop.join(mdfunischdists_totalpop, on='SchDistUGEOID', how='full', coalesce=True), how='left', on='SchDistUGEOID').pipe(calculate_stats).collect()
unischdists_totalpop.to_pandas().to_csv(f"{OUTPUTDIR}/unischdists_totalpop.csv", index=False)
ss = unischdists_totalpop.pipe(calculate_ss, geography="USD", sizecategory = "All", characteristic = "Total Population")
outputdflist.append(ss)

# Uni School Districts Total Population By Size
unischdists_totalpop = unischdists_totalpop.with_columns(Total_PopSize = pl.col('HDF_Population').cut([1000,5000,10_000,50_000,100_000], left_closed=True))
for i in unischdists_totalpop['Total_PopSize'].cat.get_categories():
    ss = unischdists_totalpop.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="USD", sizecategory = str(i), characteristic = "Total Population")
    outputdflist.append(ss)

# Minor Civil Division Total Population
hdfmcds_totalpop = dfhdf().group_by(['MCDGEOID']).agg(HDF_Population = pl.len())
mdfmcds_totalpop = dfmdf().group_by(['MCDGEOID']).agg(MDF_Population = pl.len())
mcds_totalpop =  allmcdsdf.join(hdfmcds_totalpop.join(mdfmcds_totalpop, on='MCDGEOID', how='full', coalesce=True), how='left', on='MCDGEOID').pipe(calculate_stats).collect()
mcds_totalpop.to_pandas().to_csv(f"{OUTPUTDIR}/mcds_totalpop.csv", index=False)
ss = mcds_totalpop.pipe(calculate_ss, geography="MCD", sizecategory = "All", characteristic = "Total Population")
outputdflist.append(ss)

# Minor Civil Division Total Population By Size
mcds_totalpop = mcds_totalpop.with_columns(Total_PopSize = pl.col('HDF_Population').cut([1000,5000,10_000,50_000,100_000], left_closed=True))
for i in mcds_totalpop['Total_PopSize'].cat.get_categories():
    ss = mcds_totalpop.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="MCD", sizecategory = str(i), characteristic = "Total Population")
    outputdflist.append(ss)

# Federal AIR Total Population
hdffedairs_totalpop = dfhdf().group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
mdffedairs_totalpop = dfmdf().group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
fedairs_totalpop =  allfedairsdf.join(hdffedairs_totalpop.join(mdffedairs_totalpop, on='FedAIRGEOID', how='full', coalesce=True), how='left', on='FedAIRGEOID').pipe(calculate_stats).collect()
fedairs_totalpop.to_pandas().to_csv(f"{OUTPUTDIR}/fedairs_totalpop.csv", index=False)
ss = fedairs_totalpop.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "Total Population")
outputdflist.append(ss)

# Federal AIR Total Population By Size
fedairs_totalpop = fedairs_totalpop.with_columns(Total_PopSize = pl.col('HDF_Population').cut([100,1000,10_000], left_closed=True))
for i in fedairs_totalpop['Total_PopSize'].cat.get_categories():
    ss = fedairs_totalpop.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="Fed AIR", sizecategory = str(i), characteristic = "Total Population")
    outputdflist.append(ss)

# OTSA Total Population
hdfotsas_totalpop = dfhdf().group_by(['OTSAGEOID']).agg(HDF_Population = pl.len())
mdfotsas_totalpop = dfmdf().group_by(['OTSAGEOID']).agg(MDF_Population = pl.len())
otsas_totalpop = allotsasdf.join(hdfotsas_totalpop.join(mdfotsas_totalpop, on='OTSAGEOID', how='full', coalesce=True), how='left', on='OTSAGEOID').pipe(calculate_stats).collect()
otsas_totalpop.to_pandas().to_csv(f"{OUTPUTDIR}/otsas_totalpop.csv", index=False)
ss = otsas_totalpop.pipe(calculate_ss, geography="OTSA", sizecategory = "All", characteristic = "Total Population")
outputdflist.append(ss)

# ANVSA Total Population
hdfanvsas_totalpop = dfhdf().group_by(['ANVSAGEOID']).agg(HDF_Population = pl.len())
mdfanvsas_totalpop = dfmdf().group_by(['ANVSAGEOID']).agg(MDF_Population = pl.len())
anvsas_totalpop = allanvsasdf.join(hdfanvsas_totalpop.join(mdfanvsas_totalpop, on='ANVSAGEOID', how='full', coalesce=True), how='left', on='ANVSAGEOID').pipe(calculate_stats).collect()
anvsas_totalpop.to_pandas().to_csv(f"{OUTPUTDIR}/anvsas_totalpop.csv", index=False)
ss = anvsas_totalpop.pipe(calculate_ss, geography="ANVSA", sizecategory = "All", characteristic = "Total Population")
outputdflist.append(ss)

# ANVSA Total Population By Size
anvsas_totalpop = anvsas_totalpop.with_columns(Total_PopSize = pl.col('HDF_Population').cut([100,1000,10_000], left_closed=True))
for i in anvsas_totalpop['Total_PopSize'].cat.get_categories():
    ss = anvsas_totalpop.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="ANVSA", sizecategory = str(i), characteristic = "Total Population")
    outputdflist.append(ss)

# AIANNH Areas Total Population
hdfaiannh_totalpop = dfhdf().group_by(['AIANNHGEOID']).agg(HDF_Population = pl.len())
mdfaiannh_totalpop = dfmdf().group_by(['AIANNHGEOID']).agg(MDF_Population = pl.len())
aiannh_totalpop = allaiannhdf.join(hdfaiannh_totalpop.join(mdfaiannh_totalpop, on='AIANNHGEOID', how='full', coalesce=True), how='left', on='AIANNHGEOID').pipe(calculate_stats).collect()
aiannh_totalpop.to_pandas().to_csv(f"{OUTPUTDIR}/aiannh_totalpop.csv", index=False)
ss = aiannh_totalpop.pipe(calculate_ss, geography="AIANNH Area", sizecategory = "All", characteristic = "Total Population")
outputdflist.append(ss)

# AIANNH Areas Total Population By Size
aiannh_totalpop = aiannh_totalpop.with_columns(Total_PopSize = pl.col('HDF_Population').cut([100,500,1000], left_closed=True))
for i in aiannh_totalpop['Total_PopSize'].cat.get_categories():
    ss = aiannh_totalpop.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="AIANNH Area", sizecategory = str(i), characteristic = "Total Population")
    outputdflist.append(ss)

print("{} Total Population Done".format(datetime.now()))

# County Total Population 18+
hdfcounties_totalpop18p = dfhdf().filter(pl.col('QAGE') >= 18).group_by('CountyGEOID').agg(HDF_Population = pl.len())
mdfcounties_totalpop18p = dfmdf().filter(pl.col('QAGE') >= 18).group_by('CountyGEOID').agg(MDF_Population = pl.len())
counties_totalpop18p =  allcountiesdf.join(hdfcounties_totalpop18p.join(mdfcounties_totalpop18p, on='CountyGEOID', how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
ss = counties_totalpop18p.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Total Population Aged 18+")
outputdflist.append(ss)

counties_totalpop18p = counties_totalpop18p.join(counties_totalpop.select(['CountyGEOID','Total_PopSize']), on ='CountyGEOID', how='full', coalesce=True)
for i in counties_totalpop18p['Total_PopSize'].cat.get_categories():
    ss = counties_totalpop18p.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Total Population Aged 18+")
    outputdflist.append(ss)

# Places Total Population 18+
hdfplaces_totalpop18p = dfhdf().filter(pl.col('QAGE') >= 18).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
mdfplaces_totalpop18p = dfmdf().filter(pl.col('QAGE') >= 18).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
places_totalpop18p =  allplacesdf.join(hdfplaces_totalpop18p.join(mdfplaces_totalpop18p, on='IncPlaceGEOID', how='full', coalesce=True), how='left', on='IncPlaceGEOID').pipe(calculate_stats).collect()
ss = places_totalpop18p.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "Total Population Aged 18+")
outputdflist.append(ss)

places_totalpop18p = places_totalpop18p.join(places_totalpop.select(['IncPlaceGEOID','Total_PopSize']), on ='IncPlaceGEOID', how='full', coalesce=True)
for i in places_totalpop18p['Total_PopSize'].cat.get_categories():
    ss = places_totalpop18p.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "Total Population Aged 18+")
    outputdflist.append(ss)

# Tracts Total Population 18+
hdftracts_totalpop18p = dfhdf().filter(pl.col('QAGE') >= 18).group_by('TractGEOID').size().agg(HDF_Population = pl.len())
mdftracts_totalpop18p = dfmdf().filter(pl.col('QAGE') >= 18).group_by('TractGEOID').size().agg(MDF_Population = pl.len())
tracts_totalpop18p =  alltractsdf.join(hdftracts_totalpop18p.join(mdftracts_totalpop18p, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
ss = tracts_totalpop18p.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Total Population Aged 18+")
outputdflist.append(ss)

# TODO
# if runPRhere:
#     # PR Counties/Municipios Total Population 18+
#     hdfcountiespr_totalpop18p = dfhdfpr[(dfhdfpr['QAGE'] >= 18)].group_by('CountyGEOID').agg(HDF_Population = pl.len())
#     mdfcountiespr_totalpop18p = dfmdfpr[(dfmdfpr['QAGE'] >= 18)].group_by('CountyGEOID').agg(MDF_Population = pl.len())
#     countiespr_totalpop18p =  pd.merge(hdfcountiespr_totalpop18p, mdfcountiespr_totalpop18p, on='GEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
#     ss = countiespr_totalpop18p.pipe(calculate_ss, geography="PR County/Municipio", sizecategory = "All", characteristic = "Total Population Aged 18+")
#     outputdflist.append(ss)

#     # PR Tracts Total Population 18+
#     hdftractspr_totalpop18p = dfhdfpr[(dfhdfpr['QAGE'] >= 18)].group_by('TractGEOID').agg(HDF_Population = pl.len())
#     mdftractspr_totalpop18p = dfmdfpr[(dfmdfpr['QAGE'] >= 18)].group_by('TractGEOID').agg(MDF_Population = pl.len())
#     tractspr_totalpop18p =  pd.merge(hdftractspr_totalpop18p, mdftractspr_totalpop18p, on='GEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
#     ss = tractspr_totalpop18p.pipe(calculate_ss, geography="PR Tract", sizecategory = "All", characteristic = "Total Population Aged 18+")
#     outputdflist.append(ss)

# Minor Civil Division Total Population 18+
hdfmcds_totalpop18p = dfhdf().filter(pl.col('QAGE') >= 18).group_by(['MCDGEOID']).agg(HDF_Population = pl.len())
mdfmcds_totalpop18p = dfmdf().filter(pl.col('QAGE') >= 18).group_by(['MCDGEOID']).agg(MDF_Population = pl.len())
mcds_totalpop18p =  allmcdsdf.join(hdfmcds_totalpop18p.join(mdfmcds_totalpop18p, on='MCDGEOID', how = 'full', coalesce=True), how='left', on='MCDGEOID').pipe(calculate_stats).collect()
ss = mcds_totalpop18p.pipe(calculate_ss, geography="MCD", sizecategory = "All", characteristic = "Total Population Aged 18+")
outputdflist.append(ss)

mcds_totalpop18p = mcds_totalpop18p.join(mcds_totalpop.select(['MCDGEOID','Total_PopSize']), on ='MCDGEOID', how='full', coalesce=True)
for i in mcds_totalpop18p['Total_PopSize'].cat.get_categories():
    ss = mcds_totalpop18p.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="MCD", sizecategory = str(i), characteristic = "Total Population Aged 18+")
    outputdflist.append(ss)

# Federal AIR Total Population 18+
hdffedairs_totalpop18p = dfhdf().filter(pl.col('QAGE') >= 18).group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
mdffedairs_totalpop18p = dfmdf().filter(pl.col('QAGE') >= 18).group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
fedairs_totalpop18p =  allfedairsdf.join(hdffedairs_totalpop18p.join(mdffedairs_totalpop18p, on='FedAIRGEOID', how='full', coalesce=True), how='left', on='FedAIRGEOID').pipe(calculate_stats).collect()
ss = fedairs_totalpop18p.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "Total Population Aged 18+")
outputdflist.append(ss)

fedairs_totalpop18p = fedairs_totalpop18p.join(fedairs_totalpop.select(['FedAIRGEOID','Total_PopSize']), on ='FedAIRGEOID', how='full', coalesce=True)
for i in fedairs_totalpop18p['Total_PopSize'].cat.get_categories():
    ss = fedairs_totalpop18p.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="Fed AIR", sizecategory = str(i), characteristic = "Total Population Aged 18+")
    outputdflist.append(ss)

# OTSA Total Population 18+
hdfotsas_totalpop18p = dfhdf().filter(pl.col('QAGE') >= 18).group_by(['OTSAGEOID']).agg(HDF_Population = pl.len())
mdfotsas_totalpop18p = dfmdf().filter(pl.col('QAGE') >= 18).group_by(['OTSAGEOID']).agg(MDF_Population = pl.len())
otsas_totalpop18p =  allotsasdf.join(hdfotsas_totalpop18p.join(mdfotsas_totalpop18p, on='OTSAGEOID', how='full', coalesce=True), how='left', on='OTSAGEOID').pipe(calculate_stats).collect()
ss = otsas_totalpop18p.pipe(calculate_ss, geography="OTSA", sizecategory = "All", characteristic = "Total Population Aged 18+")
outputdflist.append(ss)

# ANVSA Total Population 18+
hdfanvsas_totalpop18p = dfhdf().filter(pl.col('QAGE') >= 18).group_by(['ANVSAGEOID']).agg(HDF_Population = pl.len())
mdfanvsas_totalpop18p = dfmdf().filter(pl.col('QAGE') >= 18).group_by(['ANVSAGEOID']).agg(MDF_Population = pl.len())
anvsas_totalpop18p =  allanvsasdf.join(hdfanvsas_totalpop18p.join(mdfanvsas_totalpop18p, on='ANVSAGEOID', how='full', coalesce=True), how='left', on='ANVSAGEOID').pipe(calculate_stats).collect()
ss = anvsas_totalpop18p.pipe(calculate_ss, geography="ANVSA", sizecategory = "All", characteristic = "Total Population Aged 18+")
outputdflist.append(ss)

anvsas_totalpop18p = anvsas_totalpop18p.join(anvsas_totalpop.select(['ANVSAGEOID','Total_PopSize']), on = 'ANVSAGEOID', how='full', coalesce = True)
for i in anvsas_totalpop18p['Total_PopSize'].cat.get_categories():
    ss = anvsas_totalpop18p.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="ANVSA", sizecategory = str(i), characteristic = "Total Population Aged 18+")
    outputdflist.append(ss)

print("{} Total Population 18+ Done".format(datetime.now()))

# State Hispanic Origin
hdfstates_hisp = dfhdf().filter(pl.col('CENHISP')=='2').group_by('StateGEOID').agg(HDF_Population = pl.len())
mdfstates_hisp = dfmdf().filter(pl.col('CENHISP')=='2').group_by('StateGEOID').agg(MDF_Population = pl.len())
states_hisp = allstatesdf.join(hdfstates_hisp.join(mdfstates_hisp, on = "StateGEOID", how='full', coalesce=True), how='left', on='StateGEOID').pipe(calculate_stats).collect()
ss = states_hisp.pipe(calculate_ss, geography = "State", sizecategory = "All", characteristic = "Hispanic")
outputdflist.append(ss)

hdfstates_nonhisp = dfhdf().filter(pl.col('CENHISP')=='1').group_by('StateGEOID').agg(HDF_Population = pl.len())
mdfstates_nonhisp = dfmdf().filter(pl.col('CENHISP')=='1').group_by('StateGEOID').agg(MDF_Population = pl.len())
states_nonhisp = allstatesdf.join(hdfstates_nonhisp.join(mdfstates_nonhisp, on = "StateGEOID", how='full', coalesce=True), how='left', on='StateGEOID').pipe(calculate_stats).collect()
ss = states_nonhisp.pipe(calculate_ss, geography = "State", sizecategory = "All", characteristic = "Not Hispanic")
outputdflist.append(ss)

# County Hispanic Origin
hdfcounties_hisp = dfhdf().filter(pl.col('CENHISP')=='2').group_by('CountyGEOID').agg(HDF_Population = pl.len())
mdfcounties_hisp = dfmdf().filter(pl.col('CENHISP')=='2').group_by('CountyGEOID').agg(MDF_Population = pl.len())
counties_hisp = allcountiesdf.join(hdfcounties_hisp.join(mdfcounties_hisp, on = "CountyGEOID", how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
ss = counties_hisp.pipe(calculate_ss, geography = "County", sizecategory = "All", characteristic = "Hispanic")
outputdflist.append(ss)

hdfcounties_nonhisp = dfhdf().filter(pl.col('CENHISP')=='1').group_by(['CountyGEOID']).agg(HDF_Population = pl.len())
mdfcounties_nonhisp = dfmdf().filter(pl.col('CENHISP')=='1').group_by(['CountyGEOID']).agg(MDF_Population = pl.len())
counties_nonhisp = allcountiesdf.join(hdfcounties_nonhisp.join(mdfcounties_nonhisp, on = "CountyGEOID", how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
ss = counties_nonhisp.pipe(calculate_ss, geography = "County", sizecategory = "All", characteristic = "Not Hispanic")
outputdflist.append(ss)

counties_hisp = counties_hisp.with_columns(Hisp_PopSize = pl.col('HDF_Population').cut(breaks = [10,100], left_closed = True))
for i in counties_hisp['Hisp_PopSize'].cat.get_categories():
    ss = counties_hisp.filter(pl.col('Hisp_PopSize') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Hispanic")
    outputdflist.append(ss)

counties_nonhisp = counties_nonhisp.with_columns(NonHisp_PopSize = pl.col('HDF_Population').cut(breaks = [10,100], left_closed = True))
for i in counties_nonhisp['NonHisp_PopSize'].cat.get_categories():
    ss = counties_nonhisp.filter(pl.col('NonHisp_PopSize') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Not Hispanic")
    outputdflist.append(ss)

# Place Hispanic Origin
hdfplaces_hisp = dfhdf().filter(pl.col('CENHISP')=='2').group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
mdfplaces_hisp = dfmdf().filter(pl.col('CENHISP')=='2').group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
places_hisp = allplacesdf.join(hdfplaces_hisp.join(mdfplaces_hisp, on='IncPlaceGEOID', how='full', coalesce=True), how='left', on='IncPlaceGEOID').pipe(calculate_stats).collect()
ss = places_hisp.pipe(calculate_ss, geography = "Place", sizecategory = "All", characteristic = "Hispanic")
outputdflist.append(ss)

hdfplaces_nonhisp = dfhdf().filter(pl.col('CENHISP')=='1').group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
mdfplaces_nonhisp = dfmdf().filter(pl.col('CENHISP')=='1').group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
places_nonhisp = allplacesdf.join(hdfplaces_nonhisp.join(mdfplaces_nonhisp, on='IncPlaceGEOID', how='full', coalesce=True), how='left', on='IncPlaceGEOID').pipe(calculate_stats).collect()
ss = places_nonhisp.pipe(calculate_ss, geography = "Place", sizecategory = "All", characteristic = "Not Hispanic")
outputdflist.append(ss)

places_hisp = places_hisp.with_columns(Hisp_PopSize = pl.col('HDF_Population').cut(breaks = [10,100], left_closed = True))
for i in places_hisp['Hisp_PopSize'].cat.get_categories():
    ss = places_hisp.filter(pl.col('Hisp_PopSize') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "Hispanic")
    outputdflist.append(ss)

places_nonhisp = places_nonhisp.with_columns(Hisp_PopSize = pl.col('HDF_Population').cut(breaks = [10,100], left_closed = True))
for i in places_nonhisp['Hisp_PopSize'].cat.get_categories():
    ss = places_nonhisp.filter(pl.col('Hisp_PopSize') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "Not Hispanic")
    outputdflist.append(ss)

# Tracts Hispanic Origin
hdftracts_hisp = dfhdf().filter(pl.col('CENHISP')=='2').group_by('TractGEOID').agg(HDF_Population = pl.len())
mdftracts_hisp = dfmdf().filter(pl.col('CENHISP')=='2').group_by('TractGEOID').agg(MDF_Population = pl.len())
tracts_hisp =  alltractsdf.join(hdftracts_hisp.join(mdftracts_hisp, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
ss = tracts_hisp.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Hispanic")
outputdflist.append(ss)

hdftracts_nonhisp = dfhdf().filter(pl.col('CENHISP')=='1').group_by('TractGEOID').agg(HDF_Population = pl.len())
mdftracts_nonhisp = dfmdf().filter(pl.col('CENHISP')=='1').group_by('TractGEOID').agg(MDF_Population = pl.len())
tracts_nonhisp =  alltractsdf.join(hdftracts_nonhisp.join(mdftracts_nonhisp, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
ss = tracts_nonhisp.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Not Hispanic")
outputdflist.append(ss)

tracts_hisp = tracts_hisp.with_columns(Hisp_PopSize = pl.col('HDF_Population').cut(breaks = [10,100], left_closed = True))
for i in tracts_hisp['Hisp_PopSize'].cat.get_categories():
    ss = tracts_hisp.filter(pl.col('Hisp_PopSize') == i).pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Hispanic")
    outputdflist.append(ss)

tracts_nonhisp = tracts_nonhisp.with_columns(Hisp_PopSize = pl.col('HDF_Population').cut(breaks = [10,100], left_closed = True))
for i in tracts_nonhisp['Hisp_PopSize'].cat.get_categories():
    ss = tracts_nonhisp.filter(pl.col('Hisp_PopSize') == i).pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Not Hispanic")
    outputdflist.append(ss)

print("{} Hispanic Origin Done".format(datetime.now()))

# State Race Alone
for r in racealonecats:
    hdfstates_racealone = dfhdf().filter(pl.col('RACEALONE') == r).group_by('StateGEOID').agg(HDF_Population = pl.len())
    mdfstates_racealone = dfmdf().filter(pl.col('RACEALONE') == r).group_by('StateGEOID').agg(MDF_Population = pl.len())
    states_racealone = allstatesdf.join(hdfstates_racealone.join(mdfstates_racealone, on = "StateGEOID", how='full', coalesce=True), how='left', on='StateGEOID').pipe(calculate_stats).collect()
    ss = states_racealone.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "{race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)

# County Race Alone
for r in racealonecats:
    hdfcounties_racealone = dfhdf().filter(pl.col('RACEALONE') == r).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_racealone = dfmdf().filter(pl.col('RACEALONE') == r).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_racealone =  allcountiesdf.join(hdfcounties_racealone.join(mdfcounties_racealone, on = "CountyGEOID", how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
    ss = counties_racealone.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    counties_racealone = counties_racealone.with_columns(Race_PopSize = pl.col('HDF_Population').cut(breaks = [10,100], left_closed = True))
    for i in counties_racealone['Race_PopSize'].cat.get_categories():
        # temp = counties_racealone[counties_racealone['Race_PopSize'] == i]
        # temp.to_csv(f"{OUTPUTDIR}/counties_racealone_race{r}_{i}.csv",index=False)
        ss = counties_racealone.filter(pl.col('Race_PopSize') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "{race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)

# Place Race Alone
for r in racealonecats:
    hdfplaces_racealone = dfhdf().filter(pl.col('RACEALONE') == r).group_by('IncPlaceGEOID').agg(HDF_Population = pl.len())
    mdfplaces_racealone = dfmdf().filter(pl.col('RACEALONE') == r).group_by('IncPlaceGEOID').agg(MDF_Population = pl.len())
    places_racealone =  allplacesdf.join(hdfplaces_racealone.join(mdfplaces_racealone, on='IncPlaceGEOID', how='full', coalesce=True), how='left', on='IncPlaceGEOID').pipe(calculate_stats).collect()
    ss = places_racealone.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "{race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    places_racealone = places_racealone.with_columns(Race_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in places_racealone['Race_PopSize'].cat.get_categories():
        # temp = places_racealone[places_racealone['Race_PopSize'] == i]
        # temp.to_csv(f"{OUTPUTDIR}/places_racealone_race{r}_{i}.csv",index=False)
        ss = places_racealone.filter(pl.col('Race_PopSize') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "{race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)

# Tract Race Alone
for r in racealonecats:
    hdftracts_racealone = dfhdf().filter(pl.col('RACEALONE') == r).group_by('TractGEOID').agg(HDF_Population = pl.len())
    mdftracts_racealone = dfmdf().filter(pl.col('RACEALONE') == r).group_by('TractGEOID').agg(MDF_Population = pl.len())
    tracts_racealone =  alltractsdf.join(hdftracts_racealone.join(mdftracts_racealone, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_racealone.filter(pl.col('RACEALONE') == r).pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "{race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    tracts_racealone = tracts_racealone.with_columns(Race_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in tracts_racealone['Race_PopSize'].cat.get_categories():
        ss = tracts_racealone.filter(pl.col('Race_PopSize') == i).pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "{race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)

# ## TODO
# if runPRhere:
#     # PR Counties/Municipios Race Alone
#     for r in racealonecats:
#         hdfcountiespr_racealone = dfhdfpr.filter(pl.col('RACEALONE') == r).group_by('CountyGEOID').agg(HDF_Population = pl.len())
#         mdfcountiespr_racealone = dfmdfpr.filter(pl.col('RACEALONE') == r).group_by('CountyGEOID').agg(MDF_Population = pl.len())
#         countiespr_racealone =  allcountiesprdf.join(hdfcountiespr_racealone.join(mdfcountiespr_racealone, on = "CountyGEOID", how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
#         ss = countiespr_racealone.pipe(calculate_ss, geography="PR County/Municipio", sizecategory = "All", characteristic = "{race}".format(race = racealonedict.get(r)))
#         outputdflist.append(ss)
#         countiespr_racealone = countiespr_racealone.with_columns(Race_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
#         for i in countiespr_racealone['Race_PopSize'].cat.get_categories():
#             ss = countiespr_racealone.filter(pl.col('Race_PopSize') == i).pipe(calculate_ss, geography="PR County/Municipio", sizecategory = str(i), characteristic = "{race}".format(race = racealonedict.get(r)))
#             outputdflist.append(ss)

#     # PR Tracts Race Alone
#     for r in racealonecats:
#         hdftractspr_racealone = dfhdfpr.filter(pl.col('RACEALONE') == r).group_by(['TABBLKST', 'TABBLKCOU','TABTRACTCE']).agg(HDF_Population = pl.len())
#         mdftractspr_racealone = dfmdfpr.filter(pl.col('RACEALONE') == r).group_by(['TABBLKST', 'TABBLKCOU','TABTRACTCE']).agg(MDF_Population = pl.len())
#         tractspr_racealone =  alltractsprdf.join(hdftractspr_racealone.join(mdftractspr_racealone, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
#         ss = tractspr_racealone.pipe(calculate_ss, geography="PR Tract", sizecategory = "All", characteristic = "{race}".format(race = racealonedict.get(r)))
#         outputdflist.append(ss)
#         tractspr_racealone = tractspr_racealone.with_columns(Race_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
#         for i in tractspr_racealone['Race_PopSize'].cat.get_categories():
#             ss = tractspr_racealone.filter(pl.col('Race_PopSize') == i).pipe(calculate_ss, geography="PR Tract", sizecategory = str(i), characteristic = "{race}".format(race = racealonedict.get(r)))
#             outputdflist.append(ss)

# Federal AIR Race Alone
for r in racealonecats:
    hdffedairs_racealone = dfhdf().filter(pl.col('RACEALONE') == r).group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
    mdffedairs_racealone = dfmdf().filter(pl.col('RACEALONE') == r).group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
    fedairs_racealone =  allfedairsdf.join(hdffedairs_racealone.join(mdffedairs_racealone, on='FedAIRGEOID', how='full', coalesce=True), how='left', on='FedAIRGEOID').pipe(calculate_stats).collect()
    ss = fedairs_racealone.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "{race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    fedairs_racealone = fedairs_racealone.with_columns(Race_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in fedairs_racealone['Race_PopSize'].cat.get_categories():
        ss = fedairs_racealone.filter(pl.col('Race_PopSize') == i).pipe(calculate_ss, geography="Fed AIR", sizecategory = str(i), characteristic = "{race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)

# OTSA Race Alone
for r in racealonecats:
    hdfotsas_racealone = dfhdf().filter(pl.col('RACEALONE') == r).group_by(['OTSAGEOID']).agg(HDF_Population = pl.len())
    mdfotsas_racealone = dfmdf().filter(pl.col('RACEALONE') == r).group_by(['OTSAGEOID']).agg(MDF_Population = pl.len())
    otsas_racealone =  allotsasdf.join(hdfotsas_racealone.join(mdfotsas_racealone, on='OTSAGEOID', how='full', coalesce=True), how='left', on='OTSAGEOID').pipe(calculate_stats).collect()
    ss = otsas_racealone.pipe(calculate_ss, geography="OTSA", sizecategory = "All", characteristic = "{race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    otsas_racealone = otsas_racealone.with_columns(Race_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in otsas_racealone['Race_PopSize'].cat.get_categories():
        ss = otsas_racealone.filter(pl.col('Race_PopSize') == i).pipe(calculate_ss, geography="OTSA", sizecategory = str(i), characteristic = "{race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)

# ANVSA Race Alone
for r in racealonecats:
    hdfanvsas_racealone = dfhdf().filter(pl.col('RACEALONE') == r).group_by(['ANVSAGEOID']).agg(HDF_Population = pl.len())
    mdfanvsas_racealone = dfmdf().filter(pl.col('RACEALONE') == r).group_by(['ANVSAGEOID']).agg(MDF_Population = pl.len())
    anvsas_racealone =  allanvsasdf.join(hdfanvsas_racealone.join(mdfanvsas_racealone, on='ANVSAGEOID', how='full', coalesce=True), how='left', on='ANVSAGEOID').pipe(calculate_stats).collect()
    ss = anvsas_racealone.pipe(calculate_ss, geography="ANVSA", sizecategory = "All", characteristic = "{race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    anvsas_racealone = anvsas_racealone.with_columns(Race_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in anvsas_racealone['Race_PopSize'].cat.get_categories():
        ss = anvsas_racealone.filter(pl.col('Race_PopSize') == i).pipe(calculate_ss, geography="ANVSA", sizecategory = str(i), characteristic = "{race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)

print("{} Race Alone Done".format(datetime.now()))

# State Hispanic By Race Alone
for r in racealonecats:
    hdfstates_hispracealone = dfhdf().filter((pl.col('CENHISP') == '2'),(pl.col('RACEALONE') == r)).group_by(['TABBLKST']).agg(HDF_Population = pl.len())
    mdfstates_hispracealone = dfmdf().filter((pl.col('CENHISP') == '2'),(pl.col('RACEALONE') == r)).group_by(['TABBLKST']).agg(MDF_Population = pl.len())
    states_hispracealone =  pd.merge(hdfstates_hispracealone, mdfstates_hispracealone, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = states_hispracealone.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "Hispanic {race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    hdfstates_nonhispracealone = dfhdf().filter((pl.col('CENHISP') == '1'),(pl.col('RACEALONE') == r)).group_by(['TABBLKST']).agg(HDF_Population = pl.len())
    mdfstates_nonhispracealone = dfmdf().filter((pl.col('CENHISP') == '1'),(pl.col('RACEALONE') == r)).group_by(['TABBLKST']).agg(MDF_Population = pl.len())
    states_nonhispracealone =  pd.merge(hdfstates_nonhispracealone, mdfstates_nonhispracealone, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = states_nonhispracealone.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "Non-Hispanic {race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)


# County Hispanic By Race Alone
for r in racealonecats:
    hdfcounties_hispracealone = dfhdf().filter((pl.col('CENHISP') == '2'),(pl.col('RACEALONE') == r)).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_hispracealone = dfmdf().filter((pl.col('CENHISP') == '2'),(pl.col('RACEALONE') == r)).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_hispracealone =  allcountiesdf.join(hdfcounties_hispracealone.join(mdfcounties_hispracealone, on = "CountyGEOID", how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
    ss = counties_hispracealone.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Hispanic {race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    hdfcounties_nonhispracealone = dfhdf().filter((pl.col('CENHISP') == '1'),(pl.col('RACEALONE') == r)).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_nonhispracealone = dfmdf().filter((pl.col('CENHISP') == '1'),(pl.col('RACEALONE') == r)).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_nonhispracealone =  allcountiesdf.join(hdfcounties_nonhispracealone.join(mdfcounties_nonhispracealone, on = "CountyGEOID", how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
    ss = counties_nonhispracealone.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Non-Hispanic {race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    counties_hispracealone = counties_hispracealone.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in counties_hispracealone['HispRace_PopSize'].cat.get_categories():
        ss = counties_hispracealone[counties_hispracealone['HispRace_PopSize'] == i].pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Hispanic {race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)
    counties_nonhispracealone = counties_nonhispracealone.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in counties_nonhispracealone['HispRace_PopSize'].cat.get_categories():
        ss = counties_nonhispracealone[counties_nonhispracealone['HispRace_PopSize'] == i].pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Non-Hispanic {race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)

# Place Hispanic By Race Alone
for r in racealonecats:
    hdfplaces_hispracealone = dfhdf().filter((pl.col('CENHISP') == '2'),(pl.col('RACEALONE') == r)).group_by('IncPlaceGEOID').agg(HDF_Population = pl.len())
    mdfplaces_hispracealone = dfmdf().filter((pl.col('CENHISP') == '2'),(pl.col('RACEALONE') == r)).group_by('IncPlaceGEOID').agg(MDF_Population = pl.len())
    places_hispracealone =  allplacesdf.join(hdfplaces_hispracealone.join(mdfplaces_hispracealone, on='IncPlaceGEOID', how='full', coalesce=True), how='left', on='IncPlaceGEOID').pipe(calculate_stats).collect()
    ss = places_hispracealone.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "Hispanic {race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    hdfplaces_nonhispracealone = dfhdf().filter((pl.col('CENHISP') == '1'),(pl.col('RACEALONE') == r)).group_by('IncPlaceGEOID').agg(HDF_Population = pl.len())
    mdfplaces_nonhispracealone = dfmdf().filter((pl.col('CENHISP') == '1'),(pl.col('RACEALONE') == r)).group_by('IncPlaceGEOID').agg(MDF_Population = pl.len())
    places_nonhispracealone =  allplacesdf.join(hdfplaces_nonhispracealone.join(mdfplaces_nonhispracealone, on='IncPlaceGEOID', how='full', coalesce=True), how='left', on='IncPlaceGEOID').pipe(calculate_stats).collect()
    ss = places_nonhispracealone.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "Non-Hispanic {race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    places_hispracealone = places_hispracealone.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in places_hispracealone['HispRace_PopSize'].cat.get_categories():
        ss = places_hispracealone[places_hispracealone['HispRace_PopSize'] == i].pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "Hispanic {race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)
    places_nonhispracealone = places_nonhispracealone.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in places_nonhispracealone['HispRace_PopSize'].cat.get_categories():
        ss = places_nonhispracealone[places_nonhispracealone['HispRace_PopSize'] == i].pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "Non-Hispanic {race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)

# Tract Hispanic By Race Alone
for r in racealonecats:
    hdftracts_hispracealone = dfhdf().filter((pl.col('CENHISP') == '2'),(pl.col('RACEALONE') == r)).group_by('TractGEOID').agg(HDF_Population = pl.len())
    mdftracts_hispracealone = dfmdf().filter((pl.col('CENHISP') == '2'),(pl.col('RACEALONE') == r)).group_by('TractGEOID').agg(MDF_Population = pl.len())
    tracts_hispracealone =  alltractsdf.join(hdftracts_hispracealone.join(mdftracts_hispracealone, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_hispracealone.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Hispanic {race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    hdftracts_nonhispracealone = dfhdf().filter((pl.col('CENHISP') == '1'),(pl.col('RACEALONE') == r)).group_by('TractGEOID').agg(HDF_Population = pl.len())
    mdftracts_nonhispracealone = dfmdf().filter((pl.col('CENHISP') == '1'),(pl.col('RACEALONE') == r)).group_by('TractGEOID').agg(MDF_Population = pl.len())
    tracts_nonhispracealone =  alltractsdf.join(hdftracts_nonhispracealone.join(mdftracts_nonhispracealone, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_nonhispracealone.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Non-Hispanic {race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    tracts_hispracealone = tracts_hispracealone.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in tracts_hispracealone['HispRace_PopSize'].cat.get_categories():
        ss = tracts_hispracealone[tracts_hispracealone['HispRace_PopSize'] == i].pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Hispanic {race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)
    tracts_nonhispracealone = tracts_nonhispracealone.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in tracts_nonhispracealone['HispRace_PopSize'].cat.get_categories():
        ss = tracts_nonhispracealone[tracts_nonhispracealone['HispRace_PopSize'] == i].pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Non-Hispanic {race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)

# Tract Hispanic By Race Alone Aged 18+
for r in racealonecats:
    hdftracts_hispracealone18p = dfhdf().filter((pl.col('CENHISP') == '2'),(pl.col('QAGE') >= 18),(pl.col('RACEALONE') == r)).group_by('TractGEOID').agg(HDF_Population = pl.len())
    mdftracts_hispracealone18p = dfmdf().filter((pl.col('CENHISP') == '2'),(pl.col('QAGE') >= 18),(pl.col('RACEALONE') == r)).group_by('TractGEOID').agg(MDF_Population = pl.len())
    tracts_hispracealone18p =  alltractsdf.join(hdftracts_hispracealone18p.join(mdftracts_hispracealone18p, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_hispracealone18p.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Hispanic {race} Aged 18+".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    hdftracts_nonhispracealone18p = dfhdf().filter((pl.col('CENHISP') == '1'),(pl.col('QAGE') >= 18),(pl.col('RACEALONE') == r)).group_by('TractGEOID').agg(HDF_Population = pl.len())
    mdftracts_nonhispracealone18p = dfmdf().filter((pl.col('CENHISP') == '1'),(pl.col('QAGE') >= 18),(pl.col('RACEALONE') == r)).group_by('TractGEOID').agg(MDF_Population = pl.len())
    tracts_nonhispracealone18p =  alltractsdf.join(hdftracts_nonhispracealone18p.join(mdftracts_nonhispracealone18p, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_nonhispracealone18p.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Non-Hispanic {race} Aged 18+".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    tracts_hispracealone18p = tracts_hispracealone18p.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in tracts_hispracealone18p['HispRace_PopSize'].cat.get_categories():
        ss = tracts_hispracealone18p[tracts_hispracealone18p['HispRace_PopSize'] == i].pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Hispanic {race} Aged 18+".format(race = racealonedict.get(r)))
        outputdflist.append(ss)
    tracts_nonhispracealone18p = tracts_nonhispracealone18p.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in tracts_nonhispracealone18p['HispRace_PopSize'].cat.get_categories():
        ss = tracts_nonhispracealone18p[tracts_nonhispracealone18p['HispRace_PopSize'] == i].pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Non-Hispanic {race} Aged 18+".format(race = racealonedict.get(r)))
        outputdflist.append(ss)

# Block Group Hispanic By Race Alone Aged 18+
for r in racealonecats:
    hdfblockgroups_hispracealone18p = dfhdf().filter((pl.col('CENHISP') == '2'),(pl.col('QAGE') >= 18),(pl.col('RACEALONE') == r)).group_by('BlockGroupGEOID').agg(HDF_Population = pl.len())
    mdfblockgroups_hispracealone18p = dfmdf().filter((pl.col('CENHISP') == '2'),(pl.col('QAGE') >= 18),(pl.col('RACEALONE') == r)).group_by('BlockGroupGEOID').agg(MDF_Population = pl.len())
    blockgroups_hispracealone18p =  allblockgroupsdf.join(hdfblockgroups_hispracealone18p.join(mdfblockgroups_hispracealone18p, on='BlockGroupGEOID', how='full', coalesce=True), on='BlockGroupGEOID', how='left').pipe(calculate_stats).collect()
    ss = blockgroups_hispracealone18p.pipe(calculate_ss, geography="Block Group", sizecategory = "All", characteristic = "Hispanic {race} Aged 18+".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    hdfblockgroups_nonhispracealone18p = dfhdf().filter((pl.col('CENHISP') == '1'),(pl.col('QAGE') >= 18),(pl.col('RACEALONE') == r)).group_by('BlockGroupGEOID').agg(HDF_Population = pl.len())
    mdfblockgroups_nonhispracealone18p = dfmdf().filter((pl.col('CENHISP') == '1'),(pl.col('QAGE') >= 18),(pl.col('RACEALONE') == r)).group_by('BlockGroupGEOID').agg(MDF_Population = pl.len())
    blockgroups_nonhispracealone18p =  allblockgroupsdf.join(hdfblockgroups_nonhispracealone18p.join(mdfblockgroups_nonhispracealone18p, on='BlockGroupGEOID', how='full', coalesce=True), on='BlockGroupGEOID', how='left').pipe(calculate_stats).collect()
    ss = blockgroups_nonhispracealone18p.pipe(calculate_ss, geography="Block Group", sizecategory = "All", characteristic = "Non-Hispanic {race} Aged 18+".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    blockgroups_hispracealone18p = blockgroups_hispracealone18p.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in blockgroups_hispracealone18p['HispRace_PopSize'].cat.get_categories():
        ss = blockgroups_hispracealone18p[blockgroups_hispracealone18p['HispRace_PopSize'] == i].pipe(calculate_ss, geography="Block Group", sizecategory = str(i), characteristic = "Hispanic {race} Aged 18+".format(race = racealonedict.get(r)))
        outputdflist.append(ss)
    blockgroups_nonhispracealone18p = blockgroups_nonhispracealone18p.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in blockgroups_nonhispracealone18p['HispRace_PopSize'].cat.get_categories():
        ss = blockgroups_nonhispracealone18p[blockgroups_nonhispracealone18p['HispRace_PopSize'] == i].pipe(calculate_ss, geography="Block Group", sizecategory = str(i), characteristic = "Non-Hispanic {race} Aged 18+".format(race = racealonedict.get(r)))
        outputdflist.append(ss)

print("{} Hispanic By Race Alone Done".format(datetime.now()))

# Race Alone Or In Combination
print("{} Starting Hispanic By Race Alone Or In Combination".format(datetime.now()))

dfhdf = dfhdf().pipe(assign_racealone_or_incomb)
dfmdf = dfmdf().pipe(assign_racealone_or_incomb)

# TODO
# if runPRhere:
#     dfhdfpr = dfhdfpr.pipe(assign_racealone_or_incomb)
#     dfmdfpr = dfmdfpr.pipe(assign_racealone_or_incomb)

racegroups = ['whitealone-or-incomb', 'blackalone-or-incomb', 'aianalone-or-incomb', 'asianalone-or-incomb', 'nhopialone-or-incomb', 'soralone-or-incomb']

# State Race Alone Or In Combination/ Hispanic Race Alone Or In Combination/ Non-Hispanic Race Alone Or In Combination
for rg in racegroups:
    hdfstates_raceincomb = dfhdf().filter(pl.col(rg)==1).group_by('StateGEOID').agg(HDF_Population = pl.len())
    mdfstates_raceincomb = dfmdf().filter(pl.col(rg)==1).group_by('StateGEOID').agg(MDF_Population = pl.len())
    states_raceincomb =  allstatesdf.join(hdfstates_raceincomb.join(mdfstates_raceincomb, on = "StateGEOID", how='full', coalesce=True), how='left', on='StateGEOID').pipe(calculate_stats).collect()
    ss = states_raceincomb.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "{race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
for rg in racegroups:
    hdfstates_hispraceincomb = dfhdf().filter((pl.col('CENHISP') == '2'),(pl.col(rg)==1)).group_by('StateGEOID').agg(HDF_Population = pl.len())
    mdfstates_hispraceincomb = dfmdf().filter((pl.col('CENHISP') == '2'),(pl.col(rg)==1)).group_by('StateGEOID').agg(MDF_Population = pl.len())
    states_hispraceincomb =  allstatesdf.join(hdfstates_hispraceincomb.join(mdfstates_hispraceincomb, on = "StateGEOID", how='full', coalesce=True), how='left', on='StateGEOID').pipe(calculate_stats).collect()
    ss = states_hispraceincomb.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "Hispanic {race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
for rg in racegroups:
    hdfstates_nonhispraceincomb = dfhdf().filter((pl.col('CENHISP') == '1'),(pl.col(rg)==1)).group_by('StateGEOID').agg(HDF_Population = pl.len())
    mdfstates_nonhispraceincomb = dfmdf().filter((pl.col('CENHISP') == '1'),(pl.col(rg)==1)).group_by('StateGEOID').agg(MDF_Population = pl.len())
    states_nonhispraceincomb =  allstatesdf.join(hdfstates_nonhispraceincomb.join(mdfstates_nonhispraceincomb, on = "StateGEOID", how='full', coalesce=True), how='left', on='StateGEOID').pipe(calculate_stats).collect()
    ss = states_nonhispraceincomb.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "Non-Hispanic {race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)

# County Race Alone Or In Combination/ Hispanic Race Alone Or In Combination/ Non-Hispanic Race Alone Or In Combination
for rg in racegroups:
    hdfcounties_raceincomb = dfhdf().filter(pl.col(rg)==1).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_raceincomb = dfmdf().filter(pl.col(rg)==1).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_raceincomb =  allcountiesdf.join(hdfcounties_raceincomb.join(mdfcounties_raceincomb, on = "CountyGEOID", how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
    ss = counties_raceincomb.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
    counties_raceincomb = counties_raceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in counties_raceincomb['RaceInComb_SizeA'].cat.get_categories():
        ss = counties_raceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "{race}".format(race = raceincombdict.get(rg)))
        outputdflist.append(ss)
for rg in racegroups:
    hdfcounties_hispraceincomb = dfhdf().filter((pl.col('CENHISP') == '2'),(pl.col(rg)==1)).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_hispraceincomb = dfmdf().filter((pl.col('CENHISP') == '2'),(pl.col(rg)==1)).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_hispraceincomb =  allcountiesdf.join(hdfcounties_hispraceincomb.join(mdfcounties_hispraceincomb, on = "CountyGEOID", how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
    ss = counties_hispraceincomb.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Hispanic {race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
    counties_hispraceincomb = counties_hispraceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in counties_hispraceincomb['RaceInComb_SizeA'].cat.get_categories():
        ss = counties_hispraceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Hispanic {race}".format(race = raceincombdict.get(rg)))
        outputdflist.append(ss)
for rg in racegroups:
    hdfcounties_nonhispraceincomb = dfhdf().filter((pl.col('CENHISP') == '1'),(pl.col(rg)==1)).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_nonhispraceincomb = dfmdf().filter((pl.col('CENHISP') == '1'),(pl.col(rg)==1)).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_nonhispraceincomb =  allcountiesdf.join(hdfcounties_nonhispraceincomb.join(mdfcounties_nonhispraceincomb, on = "CountyGEOID", how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
    ss = counties_nonhispraceincomb.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Non-Hispanic {race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
    counties_nonhispraceincomb = counties_nonhispraceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in counties_nonhispraceincomb['RaceInComb_SizeA'].cat.get_categories():
        ss = counties_nonhispraceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Non-Hispanic {race}".format(race = raceincombdict.get(rg)))
        outputdflist.append(ss)

# Place Race Alone Or In Combination/ Hispanic Race Alone Or In Combination/ Non-Hispanic Race Alone Or In Combination
for rg in racegroups:
    hdfplaces_raceincomb = dfhdf().filter(pl.col(rg)==1).group_by('IncPlaceGEOID').agg(HDF_Population = pl.len())
    mdfplaces_raceincomb = dfmdf().filter(pl.col(rg)==1).group_by('IncPlaceGEOID').agg(MDF_Population = pl.len())
    places_raceincomb =  allplacesdf.join(hdfplaces_raceincomb.join(mdfplaces_raceincomb, on='IncPlaceGEOID', how='full', coalesce=True), how='left', on='IncPlaceGEOID').pipe(calculate_stats).collect()
    ss = places_raceincomb.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "{race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
    places_raceincomb = places_raceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in places_raceincomb['RaceInComb_SizeA'].cat.get_categories():
        ss = places_raceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "{race}".format(race = raceincombdict.get(rg)))
        outputdflist.append(ss)
for rg in racegroups:
    hdfplaces_hispraceincomb = dfhdf().filter((pl.col('CENHISP') == '2'),(pl.col(rg)==1)).group_by('IncPlaceGEOID').agg(HDF_Population = pl.len())
    mdfplaces_hispraceincomb = dfmdf().filter((pl.col('CENHISP') == '2'),(pl.col(rg)==1)).group_by('IncPlaceGEOID').agg(MDF_Population = pl.len())
    places_hispraceincomb =  allplacesdf.join(hdfplaces_hispraceincomb.join(mdfplaces_hispraceincomb, on='IncPlaceGEOID', how='full', coalesce=True), how='left', on='IncPlaceGEOID').pipe(calculate_stats).collect()
    ss = places_hispraceincomb.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "Hispanic {race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
    places_hispraceincomb = places_hispraceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in places_hispraceincomb['RaceInComb_SizeA'].cat.get_categories():
        ss = places_hispraceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "Hispanic {race}".format(race = raceincombdict.get(rg)))
        outputdflist.append(ss)
for rg in racegroups:
    hdfplaces_nonhispraceincomb = dfhdf().filter((pl.col('CENHISP') == '1'),(pl.col(rg)==1)).group_by('IncPlaceGEOID').agg(HDF_Population = pl.len())
    mdfplaces_nonhispraceincomb = dfmdf().filter((pl.col('CENHISP') == '1'),(pl.col(rg)==1)).group_by('IncPlaceGEOID').agg(MDF_Population = pl.len())
    places_nonhispraceincomb =  allplacesdf.join(hdfplaces_nonhispraceincomb.join(mdfplaces_nonhispraceincomb, on='IncPlaceGEOID', how='full', coalesce=True), how='left', on='IncPlaceGEOID').pipe(calculate_stats).collect()
    ss = places_nonhispraceincomb.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "Non-Hispanic {race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
    places_nonhispraceincomb = places_nonhispraceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in places_nonhispraceincomb['RaceInComb_SizeA'].cat.get_categories():
        ss = places_nonhispraceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "Non-Hispanic {race}".format(race = raceincombdict.get(rg)))
        outputdflist.append(ss)

# Tract Race Alone Or In Combination/ Hispanic Race Alone Or In Combination/ Non-Hispanic Race Alone Or In Combination
for rg in racegroups:
    hdftracts_raceincomb = dfhdf().filter(pl.col(rg)==1).group_by('TractGEOID').agg(HDF_Population = pl.len())
    mdftracts_raceincomb = dfmdf().filter(pl.col(rg)==1).group_by('TractGEOID').agg(MDF_Population = pl.len())
    tracts_raceincomb =  alltractsdf.join(hdftracts_raceincomb.join(mdftracts_raceincomb, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_raceincomb.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "{race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
    tracts_raceincomb = tracts_raceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in tracts_raceincomb['RaceInComb_SizeA'].cat.get_categories():
        ss = tracts_raceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "{race}".format(race = raceincombdict.get(rg)))
        outputdflist.append(ss)
for rg in racegroups:
    hdftracts_hispraceincomb = dfhdf().filter((pl.col('CENHISP') == '2'),(pl.col(rg)==1)).group_by('TractGEOID').agg(HDF_Population = pl.len())
    mdftracts_hispraceincomb = dfmdf().filter((pl.col('CENHISP') == '2'),(pl.col(rg)==1)).group_by('TractGEOID').agg(MDF_Population = pl.len())
    tracts_hispraceincomb =  alltractsdf.join(hdftracts_hispraceincomb.join(mdftracts_hispraceincomb, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_hispraceincomb.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Hispanic {race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
    tracts_hispraceincomb = tracts_hispraceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in tracts_hispraceincomb['RaceInComb_SizeA'].cat.get_categories():
        ss = tracts_hispraceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Hispanic {race}".format(race = raceincombdict.get(rg)))
        outputdflist.append(ss)
for rg in racegroups:
    hdftracts_nonhispraceincomb = dfhdf().filter((pl.col('CENHISP') == '1'),(pl.col(rg)==1)).group_by('TractGEOID').agg(HDF_Population = pl.len())
    mdftracts_nonhispraceincomb = dfmdf().filter((pl.col('CENHISP') == '1'),(pl.col(rg)==1)).group_by('TractGEOID').agg(MDF_Population = pl.len())
    tracts_nonhispraceincomb =  alltractsdf.join(hdftracts_nonhispraceincomb.join(mdftracts_nonhispraceincomb, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_nonhispraceincomb.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Non-Hispanic {race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
    tracts_nonhispraceincomb = tracts_nonhispraceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in tracts_nonhispraceincomb['RaceInComb_SizeA'].cat.get_categories():
        ss = tracts_nonhispraceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Non-Hispanic {race}".format(race = raceincombdict.get(rg)))
        outputdflist.append(ss)

#TODO
# if runPRhere:
#     # PR Counties/Municipios Race Alone Or In Combination
#     for rg in racegroups:
#         hdfcountiespr_raceincomb = dfhdfpr().filter(pl.col(rg)==1).group_by('CountyGEOID').agg(HDF_Population = pl.len())
#         mdfcountiespr_raceincomb = dfmdfpr().filter(pl.col(rg)==1).group_by('CountyGEOID').agg(MDF_Population = pl.len())
#         countiespr_raceincomb =  allcountiesprdf.join(hdfcountiespr_raceincomb.join(mdfcountiespr_raceincomb, on = "CountyGEOID", how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
#         ss = countiespr_raceincomb.pipe(calculate_ss, geography="PR County/Municipio", sizecategory = "All", characteristic = "{race}".format(race = raceincombdict.get(rg)))
#         outputdflist.append(ss)
#         countiespr_raceincomb = countiespr_raceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
#         for i in countiespr_raceincomb['RaceInComb_SizeA'].cat.get_categories():
#             ss = countiespr_raceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="PR County/Municipio", sizecategory = str(i), characteristic = "{race}".format(race = raceincombdict.get(rg)))
#             outputdflist.append(ss)

#     # PR Tracts Race Alone Or In Combination
#     for rg in racegroups:
#         hdftractspr_raceincomb = dfhdfpr().filter(pl.col(rg)==1).group_by('TractGEOID').agg(HDF_Population = pl.len())
#         mdftractspr_raceincomb = dfmdfpr().filter(pl.col(rg)==1).group_by('TractGEOID').agg(MDF_Population = pl.len())
#         tractspr_raceincomb =  alltractsprdf.join(hdftractspr_raceincomb.join(mdftractspr_raceincomb, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
#         ss = tractspr_raceincomb.pipe(calculate_ss, geography="PR Tract", sizecategory = "All", characteristic = "{race}".format(race = raceincombdict.get(rg)))
#         outputdflist.append(ss)
#         tractspr_raceincomb = tractspr_raceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
#         for i in tractspr_raceincomb['RaceInComb_SizeA'].cat.get_categories():
#             ss = tractspr_raceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="PR Tract", sizecategory = str(i), characteristic = "{race}".format(race = raceincombdict.get(rg)))
#             outputdflist.append(ss)

# Federal AIR Race Alone Or In Combination
for rg in racegroups:
    hdffedairs_raceincomb = dfhdf().filter(pl.col(rg) == 1).group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
    mdffedairs_raceincomb = dfmdf().filter(pl.col(rg) == 1).group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
    fedairs_raceincomb =  allfedairsdf.join(hdffedairs_raceincomb.join(mdffedairs_raceincomb, on='FedAIRGEOID', how='full', coalesce=True), how='left', on='FedAIRGEOID').pipe(calculate_stats).collect()
    ss = fedairs_raceincomb.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "{race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
    fedairs_raceincomb = fedairs_raceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in fedairs_raceincomb['RaceInComb_SizeA'].cat.get_categories():
        ss = fedairs_raceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="Fed AIR", sizecategory = str(i), characteristic = "{race}".format(race = raceincombdict.get(rg)))
        outputdflist.append(ss)

# OTSA Race Alone Or In Combination
for rg in racegroups:
    hdfotsas_raceincomb = dfhdf().filter(pl.col(rg) == 1).group_by(['OTSAGEOID']).agg(HDF_Population = pl.len())
    mdfotsas_raceincomb = dfmdf().filter(pl.col(rg) == 1).group_by(['OTSAGEOID']).agg(MDF_Population = pl.len())
    otsas_raceincomb =  allotsasdf.join(hdfotsas_raceincomb.join(mdfotsas_raceincomb, on='OTSAGEOID', how='full', coalesce=True), how='left', on='OTSAGEOID').pipe(calculate_stats).collect()
    ss = otsas_raceincomb.pipe(calculate_ss, geography="OTSA", sizecategory = "All", characteristic = "{race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
    otsas_raceincomb = otsas_raceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in otsas_raceincomb['RaceInComb_SizeA'].cat.get_categories():
        ss = otsas_raceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="OTSA", sizecategory = str(i), characteristic = "{race}".format(race = raceincombdict.get(rg)))
        outputdflist.append(ss)

# ANVSA Race Alone Or In Combination
for rg in racegroups:
    hdfanvsas_raceincomb = dfhdf().filter(pl.col(rg) == 1).group_by(['ANVSAGEOID']).agg(HDF_Population = pl.len())
    mdfanvsas_raceincomb = dfmdf().filter(pl.col(rg) == 1).group_by(['ANVSAGEOID']).agg(MDF_Population = pl.len())
    anvsas_raceincomb =  allanvsasdf.join(hdfanvsas_raceincomb.join(mdfanvsas_raceincomb, on='ANVSAGEOID', how='full', coalesce=True), how='left', on='ANVSAGEOID').pipe(calculate_stats).collect()
    ss = anvsas_raceincomb.pipe(calculate_ss, geography="ANVSA", sizecategory = "All", characteristic = "{race}".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
    anvsas_raceincomb = anvsas_raceincomb.with_columns(RaceInComb_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in anvsas_raceincomb['RaceInComb_SizeA'].cat.get_categories():
        ss = anvsas_raceincomb.filter(pl.col('RaceInComb_SizeA') == i).pipe(calculate_ss, geography="ANVSA", sizecategory = str(i), characteristic = "{race}".format(race = raceincombdict.get(rg)))
        outputdflist.append(ss)

# Tract Race Alone Or In Combination Aged 18+
for rg in racegroups:
    hdftracts_hispraceincomb18p = dfhdf().filter((pl.col('QAGE') >= 18),(pl.col('CENHISP') == '2'),(pl.col(rg)==1)).group_by('TractGEOID').agg(HDF_Population = pl.len())
    mdftracts_hispraceincomb18p = dfmdf().filter((pl.col('QAGE') >= 18),(pl.col('CENHISP') == '2'),(pl.col(rg)==1)).group_by('TractGEOID').agg(MDF_Population = pl.len())
    tracts_hispraceincomb18p =  alltractsdf.join(hdftracts_hispraceincomb18p.join(mdftracts_hispraceincomb18p, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_hispraceincomb18p.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Hispanic {race} Aged 18+".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
for rg in racegroups:
    hdftracts_nonhispraceincomb18p = dfhdf().filter((pl.col('QAGE') >= 18),(pl.col('CENHISP') == '1'),(pl.col(rg)==1)).group_by('TractGEOID').agg(HDF_Population = pl.len())
    mdftracts_nonhispraceincomb18p = dfmdf().filter((pl.col('QAGE') >= 18),(pl.col('CENHISP') == '1'),(pl.col(rg)==1)).group_by('TractGEOID').agg(MDF_Population = pl.len())
    tracts_nonhispraceincomb18p =  alltractsdf.join(hdftracts_nonhispraceincomb18p.join(mdftracts_nonhispraceincomb18p, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_nonhispraceincomb18p.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Non-Hispanic {race} Aged 18+".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)

# Block Group Race Alone Or In Combination Aged 18+
for rg in racegroups:
    hdfblockgroups_hispraceincomb18p = dfhdf().filter((pl.col('QAGE') >= 18),(pl.col('CENHISP') == '2'),(pl.col(rg)==1)).group_by('BlockGroupGEOID').agg(HDF_Population = pl.len())
    mdfblockgroups_hispraceincomb18p = dfmdf().filter((pl.col('QAGE') >= 18),(pl.col('CENHISP') == '2'),(pl.col(rg)==1)).group_by('BlockGroupGEOID').agg(MDF_Population = pl.len())
    blockgroups_hispraceincomb18p = allblockgroupsdf.join(hdfblockgroups_hispraceincomb18p.join(mdfblockgroups_hispraceincomb18p, on='BlockGroupGEOID', how='full', coalesce=True), on='BlockGroupGEOID', how='left').pipe(calculate_stats).collect()
    ss = blockgroups_hispraceincomb18p.pipe(calculate_ss, geography="Block Group", sizecategory = "All", characteristic = "Hispanic {race} Aged 18+".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)
for rg in racegroups:
    hdfblockgroups_nonhispraceincomb18p = dfhdf().filter((pl.col('QAGE') >= 18),(pl.col('CENHISP') == '1'),(pl.col(rg)==1)).group_by('BlockGroupGEOID').agg(HDF_Population = pl.len())
    mdfblockgroups_nonhispraceincomb18p = dfmdf().filter((pl.col('QAGE') >= 18),(pl.col('CENHISP') == '1'),(pl.col(rg)==1)).group_by('BlockGroupGEOID').agg(MDF_Population = pl.len())
    blockgroups_nonhispraceincomb18p = allblockgroupsdf.join(hdfblockgroups_nonhispraceincomb18p.join(mdfblockgroups_nonhispraceincomb18p, on='BlockGroupGEOID', how='full', coalesce=True), on='BlockGroupGEOID', how='left').pipe(calculate_stats).collect()
    ss = blockgroups_nonhispraceincomb18p.pipe(calculate_ss, geography="Block Group", sizecategory = "All", characteristic = "Non-Hispanic {race} Aged 18+".format(race = raceincombdict.get(rg)))
    outputdflist.append(ss)

print("{} Race Alone Or In Combination (also by Hispanic) Done".format(datetime.now()))
print("{} Starting Number of Races".format(datetime.now()))

# State Number of Races/ Hispanic Number of Races/Non-Hispanic Number of Races
for n in numracescats:
    hdfstates_numraces = dfhdf.filter(pl.col('NUMRACES') == n).group_by('StateGEOID').agg(HDF_Population = pl.len())
    mdfstates_numraces = dfmdf.filter(pl.col('NUMRACES') == n).group_by('StateGEOID').agg(MDF_Population = pl.len())
    states_numraces =  allstatesdf.join(hdfstates_numraces.join(mdfstates_numraces, on = "StateGEOID", how='full', coalesce=True), how='left', on='StateGEOID').pipe(calculate_stats).collect()
    ss = states_numraces.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "{} Race(s)".format(n))
    outputdflist.append(ss)
for n in numracescats:
    hdfstates_hispnumraces = dfhdf.filter((pl.col('CENHISP') == '2'),(pl.col('NUMRACES') == n)).group_by('StateGEOID').agg(HDF_Population = pl.len())
    mdfstates_hispnumraces = dfmdf.filter((pl.col('CENHISP') == '2'), (pl.col('NUMRACES') == n)).group_by('StateGEOID').agg(MDF_Population = pl.len())
    states_hispnumraces =  allstatesdf.join(hdfstates_hispnumraces.join(mdfstates_hispnumraces, on = "StateGEOID", how='full', coalesce=True), how='left', on='StateGEOID').pipe(calculate_stats).collect()
    ss = states_hispnumraces.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "Hispanic {} Race(s)".format(n))
    outputdflist.append(ss)
for n in numracescats:
    hdfstates_nonhispnumraces = dfhdf.filter((pl.col('CENHISP') == '1'),(pl.col('NUMRACES') == n)).group_by('StateGEOID').agg(HDF_Population = pl.len())
    mdfstates_nonhispnumraces = dfmdf.filter((pl.col('CENHISP') == '1'), (pl.col('NUMRACES') == n)).group_by('StateGEOID').agg(MDF_Population = pl.len())
    states_nonhispnumraces =  allstatesdf.join(hdfstates_nonhispnumraces.join(mdfstates_nonhispnumraces, on = "StateGEOID", how='full', coalesce=True), how='left', on='StateGEOID').pipe(calculate_stats).collect()
    ss = states_nonhispnumraces.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "Non-Hispanic {} Race(s)".format(n))
    outputdflist.append(ss)

# County Number of Races/ Hispanic Number of Races/Non-Hispanic Number of Races
for n in numracescats:
    hdfcounties_numraces = dfhdf.filter(pl.col('NUMRACES') == n).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_numraces = dfmdf.filter(pl.col('NUMRACES') == n).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_numraces =  allcountiesdf.join(hdfcounties_numraces.join(mdfcounties_numraces, on = "CountyGEOID", how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
    ss = counties_numraces.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{} Race(s)".format(n))
    outputdflist.append(ss)
    counties_numraces = counties_numraces.with_columns(NumRaces_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in counties_numraces['NumRaces_SizeA'].cat.get_categories():
        ss = counties_numraces.filter(pl.col('NumRaces_SizeA') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "{} Race(s)".format(n))
        outputdflist.append(ss)
for n in numracescats:
    hdfcounties_hispnumraces = dfhdf.filter((pl.col('CENHISP') == '2'),(pl.col('NUMRACES') == n)).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_hispnumraces = dfmdf.filter((pl.col('CENHISP') == '2'), (pl.col('NUMRACES') == n)).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_hispnumraces =  allcountiesdf.join(hdfcounties_hispnumraces.join(mdfcounties_hispnumraces, on = "CountyGEOID", how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
    ss = counties_hispnumraces.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Hispanic {} Race(s)".format(n))
    outputdflist.append(ss)
    counties_hispnumraces = counties_hispnumraces.with_columns(NumRaces_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in counties_hispnumraces['NumRaces_SizeA'].cat.get_categories():
        ss = counties_hispnumraces.filter(pl.col('NumRaces_SizeA') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Hispanic {} Race(s)".format(n))
        outputdflist.append(ss)
for n in numracescats:
    hdfcounties_nonhispnumraces = dfhdf.filter((pl.col('CENHISP') == '1'),(pl.col('NUMRACES') == n)).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_nonhispnumraces = dfmdf.filter((pl.col('CENHISP') == '1'), (pl.col('NUMRACES') == n)).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_nonhispnumraces =  allcountiesdf.join(hdfcounties_nonhispnumraces.join(mdfcounties_nonhispnumraces, on = "CountyGEOID", how='full', coalesce=True), how='left', on='CountyGEOID').pipe(calculate_stats).collect()
    ss = counties_nonhispnumraces.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Non-Hispanic {} Race(s)".format(n))
    outputdflist.append(ss)
    counties_nonhispnumraces = counties_nonhispnumraces.with_columns(NumRaces_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in counties_nonhispnumraces['NumRaces_SizeA'].cat.get_categories():
        ss = counties_nonhispnumraces.filter(pl.col('NumRaces_SizeA') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Non-Hispanic {} Race(s)".format(n))
        outputdflist.append(ss)

# Place Number of Races/ Hispanic Number of Races/Non-Hispanic Number of Races
for n in numracescats:
    hdfplaces_numraces = dfhdf.filter(pl.col('NUMRACES') == n).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_numraces = dfmdf.filter(pl.col('NUMRACES') == n).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_numraces =  allplacesdf.join(hdfplaces_numraces.join(mdfplaces_numraces, on='IncPlaceGEOID', how='full', coalesce=True), how='left', on='IncPlaceGEOID').pipe(calculate_stats).collect()
    ss = places_numraces.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "{} Race(s)".format(n))
    outputdflist.append(ss)
    places_numraces = places_numraces.with_columns(NumRaces_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in places_numraces['NumRaces_SizeA'].cat.get_categories():
        ss = places_numraces.filter(pl.col('NumRaces_SizeA') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "{} Race(s)".format(n))
        outputdflist.append(ss)
for n in numracescats:
    hdfplaces_hispnumraces = dfhdf.filter((pl.col('CENHISP') == '2'),(pl.col('NUMRACES') == n)).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_hispnumraces = dfmdf.filter((pl.col('CENHISP') == '2'), (pl.col('NUMRACES') == n)).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_hispnumraces =  allplacesdf.join(hdfplaces_hispnumraces.join(mdfplaces_hispnumraces, on='IncPlaceGEOID', how='full', coalesce=True), how='left', on='IncPlaceGEOID').pipe(calculate_stats).collect()
    ss = places_hispnumraces.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "Hispanic {} Race(s)".format(n))
    outputdflist.append(ss)
    places_hispnumraces = places_hispnumraces.with_columns(NumRaces_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in places_hispnumraces['NumRaces_SizeA'].cat.get_categories():
        ss = places_hispnumraces.filter(pl.col('NumRaces_SizeA') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "Hispanic {} Race(s)".format(n))
        outputdflist.append(ss)
for n in numracescats:
    hdfplaces_nonhispnumraces = dfhdf.filter((pl.col('CENHISP') == '1'),(pl.col('NUMRACES') == n)).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_nonhispnumraces = dfmdf.filter((pl.col('CENHISP') == '1'), (pl.col('NUMRACES') == n)).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_nonhispnumraces =  allplacesdf.join(hdfplaces_nonhispnumraces.join(mdfplaces_nonhispnumraces, on='IncPlaceGEOID', how='full', coalesce=True), how='left', on='IncPlaceGEOID').pipe(calculate_stats).collect()
    ss = places_nonhispnumraces.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "Non-Hispanic {} Race(s)".format(n))
    outputdflist.append(ss)
    places_nonhispnumraces = places_nonhispnumraces.with_columns(NumRaces_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in places_nonhispnumraces['NumRaces_SizeA'].cat.get_categories():
        ss = places_nonhispnumraces.filter(pl.col('NumRaces_SizeA') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "Non-Hispanic {} Race(s)".format(n))
        outputdflist.append(ss)

# Tract Number of Races/ Hispanic Number of Races/Non-Hispanic Number of Races
for n in numracescats:
    hdftracts_numraces = dfhdf.filter(pl.col('NUMRACES') == n).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
    mdftracts_numraces = dfmdf.filter(pl.col('NUMRACES') == n).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
    tracts_numraces =  alltractsdf.join(hdftracts_numraces.join(mdftracts_numraces, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_numraces.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "{} Race(s)".format(n))
    outputdflist.append(ss)
    tracts_numraces = tracts_numraces.with_columns(NumRaces_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in tracts_numraces['NumRaces_SizeA'].cat.get_categories():
        ss = tracts_numraces.filter(pl.col('NumRaces_SizeA') == i).pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "{} Race(s)".format(n))
        outputdflist.append(ss)
for n in numracescats:
    hdftracts_hispnumraces = dfhdf.filter((pl.col('CENHISP') == '2'),(pl.col('NUMRACES') == n)).group_by(['TABBLKST', 'TABBLKCOU','TABTRACTCE']).agg(HDF_Population = pl.len())
    mdftracts_hispnumraces = dfmdf.filter((pl.col('CENHISP') == '2'), (pl.col('NUMRACES') == n)).group_by(['TABBLKST', 'TABBLKCOU','TABTRACTCE']).agg(MDF_Population = pl.len())
    tracts_hispnumraces =  alltractsdf.join(hdftracts_hispnumraces.join(mdftracts_hispnumraces, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_hispnumraces.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Hispanic {} Race(s)".format(n))
    outputdflist.append(ss)
    tracts_hispnumraces = tracts_hispnumraces.with_columns(NumRaces_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in tracts_hispnumraces['NumRaces_SizeA'].cat.get_categories():
        ss = tracts_hispnumraces.filter(pl.col('NumRaces_SizeA') == i).pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Hispanic {} Race(s)".format(n))
        outputdflist.append(ss)
for n in numracescats:
    hdftracts_nonhispnumraces = dfhdf.filter((pl.col('CENHISP') == '1'),(pl.col('NUMRACES') == n)).group_by(['TABBLKST', 'TABBLKCOU','TABTRACTCE']).agg(HDF_Population = pl.len())
    mdftracts_nonhispnumraces = dfmdf.filter((pl.col('CENHISP') == '1'), (pl.col('NUMRACES') == n)).group_by(['TABBLKST', 'TABBLKCOU','TABTRACTCE']).agg(MDF_Population = pl.len())
    tracts_nonhispnumraces =  alltractsdf.join(hdftracts_nonhispnumraces.join(mdftracts_nonhispnumraces, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_nonhispnumraces.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Non-Hispanic {} Race(s)".format(n))
    outputdflist.append(ss)
    tracts_nonhispnumraces = tracts_nonhispnumraces.with_columns(NumRaces_SizeA = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in tracts_nonhispnumraces['NumRaces_SizeA'].cat.get_categories():
        ss = tracts_nonhispnumraces.filter(pl.col('NumRaces_SizeA') == i).pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Non-Hispanic {} Race(s)".format(n))
        outputdflist.append(ss)

# Tract Hispanic Number of Races/Non-Hispanic Number of Races Aged 18+
for n in numracescats:
    hdftracts_hispnumraces18p = dfhdf.filter((pl.col('QAGE') >= 18), (pl.col('NUMRACES') == n), (pl.col('CENHISP') == '2')).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
    mdftracts_hispnumraces18p = dfmdf.filter((pl.col('QAGE') >= 18), (pl.col('NUMRACES') == n), (pl.col('CENHISP') == '2')).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
    tracts_hispnumraces18p =  alltractsdf.join(hdftracts_hispnumraces18p.join(mdftracts_hispnumraces18p, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_hispnumraces18p.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Hispanic {} Race(s) Aged 18+".format(n))
    outputdflist.append(ss)
    hdftracts_nonhispnumraces18p = dfhdf.filter((pl.col('QAGE') >= 18), (pl.col('NUMRACES') == n), (pl.col('CENHISP') == '1')).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
    mdftracts_nonhispnumraces18p = dfmdf.filter((pl.col('QAGE') >= 18), (pl.col('NUMRACES') == n), (pl.col('CENHISP') == '1')).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
    tracts_nonhispnumraces18p =  alltractsdf.join(hdftracts_nonhispnumraces18p.join(mdftracts_nonhispnumraces18p, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
    ss = tracts_nonhispnumraces18p.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Non-Hispanic {} Race(s) Aged 18+".format(n))
    outputdflist.append(ss)

# Block Group Hispanic Number of Races/Non-Hispanic Number of Races Aged 18+
for n in numracescats:
    hdfblockgroups_hispnumraces18p = dfhdf.filter((pl.col('QAGE') >= 18), (pl.col('NUMRACES') == n), (pl.col('CENHISP') == '2')).group_by('BlockGroupGEOID').agg(HDF_Population = pl.len())
    mdfblockgroups_hispnumraces18p = dfmdf.filter((pl.col('QAGE') >= 18), (pl.col('NUMRACES') == n), (pl.col('CENHISP') == '2')).group_by('BlockGroupGEOID').agg(MDF_Population = pl.len())
    blockgroups_hispnumraces18p =  allblockgroupsdf.join(hdfblockgroups_hispnumraces18p.join(mdfblockgroups_hispnumraces18p, on='BlockGroupGEOID', how='full', coalesce=True), on='BlockGroupGEOID', how='left').pipe(calculate_stats).collect()
    ss = blockgroups_hispnumraces18p.pipe(calculate_ss, geography="Block Group", sizecategory = "All", characteristic = "Hispanic {} Race(s) Aged 18+".format(n))
    outputdflist.append(ss)
    hdfblockgroups_nonhispnumraces18p = dfhdf.filter((pl.col('QAGE') >= 18), (pl.col('NUMRACES') == n), (pl.col('CENHISP') == '1')).group_by('BlockGroupGEOID').agg(HDF_Population = pl.len())
    mdfblockgroups_nonhispnumraces18p = dfmdf.filter((pl.col('QAGE') >= 18), (pl.col('NUMRACES') == n), (pl.col('CENHISP') == '1')).group_by('BlockGroupGEOID').agg(MDF_Population = pl.len())
    blockgroups_nonhispnumraces18p =  allblockgroupsdf.join(hdfblockgroups_nonhispnumraces18p.join(mdfblockgroups_nonhispnumraces18p, on='BlockGroupGEOID', how='full', coalesce=True), on='BlockGroupGEOID', how='left').pipe(calculate_stats).collect()
    ss = blockgroups_nonhispnumraces18p.pipe(calculate_ss, geography="Block Group", sizecategory = "All", characteristic = "Non-Hispanic {} Race(s) Aged 18+".format(n))
    outputdflist.append(ss)
print("{} Num Races Done".format(datetime.now()))

## HERE

print("{} Start Detailed Age".format(datetime.now()))
# County Sex by 3 Age Groups
for g in qage_3g_cats:
    hdfcounties_3gage = dfhdf[dfhdf['QAGE_3G'] == g].group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_3gage = dfmdf[dfmdf['QAGE_3G'] == g].group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_3gage =  pd.merge(hdfcounties_3gage, mdfcounties_3gage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = counties_3gage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Age {age}".format(age=g))
    outputdflist.append(ss)
for s in sexcats:
    hdfcounties_sex = dfhdf[dfhdf['QSEX'] == s].group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_sex = dfmdf[dfmdf['QSEX'] == s].group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_sex =  pd.merge(hdfcounties_sex, mdfcounties_sex, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = counties_sex.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
for s in sexcats:
    for g in qage_3g_cats:
        hdfcounties_sex3gage = dfhdf[(dfhdf['QAGE_3G'] == g)&(dfhdf['QSEX'] == s)].group_by('CountyGEOID').agg(HDF_Population = pl.len())
        mdfcounties_sex3gage = dfmdf[(dfmdf['QAGE_3G'] == g)&(dfmdf['QSEX'] == s)].group_by('CountyGEOID').agg(MDF_Population = pl.len())
        counties_sex3gage =  pd.merge(hdfcounties_sex3gage, mdfcounties_sex3gage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = counties_sex3gage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

print("{} County Sex by 3 Age Done".format(datetime.now()))

# County Sex by 3 Age Groups [0.0, 1000.0)]
# Must use .reindex(counties_lt1000index, fill_value=0).reset_index()
for g in qage_3g_cats:
    hdfcounties_3gage = dfhdf[dfhdf['QAGE_3G'] == g].group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_3gage = dfmdf[dfmdf['QAGE_3G'] == g].group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_3gage =  pd.merge(hdfcounties_3gage, mdfcounties_3gage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = counties_3gage.pipe(calculate_ss, geography="County", sizecategory = "[0.0, 1000.0)", characteristic = "Age {age}".format(age=g))
    outputdflist.append(ss)
for s in sexcats:
    hdfcounties_sex = dfhdf[dfhdf['QSEX'] == s].group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_sex = dfmdf[dfmdf['QSEX'] == s].group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_sex =  pd.merge(hdfcounties_sex, mdfcounties_sex, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = counties_sex.pipe(calculate_ss, geography="County", sizecategory = "[0.0, 1000.0)", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
for s in sexcats:
    for g in qage_3g_cats:
        hdfcounties_sex3gage = dfhdf[(dfhdf['QAGE_3G'] == g)&(dfhdf['QSEX'] == s)].group_by('CountyGEOID').agg(HDF_Population = pl.len())
        mdfcounties_sex3gage = dfmdf[(dfmdf['QAGE_3G'] == g)&(dfmdf['QSEX'] == s)].group_by('CountyGEOID').agg(MDF_Population = pl.len())
        counties_sex3gage =  pd.merge(hdfcounties_sex3gage, mdfcounties_sex3gage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = counties_sex3gage.pipe(calculate_ss, geography="County", sizecategory = "[0.0, 1000.0)", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

# Place Sex by 3 Age Groups
for g in qage_3g_cats:
    hdfplaces_3gage = dfhdf[dfhdf['QAGE_3G'] == g].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_3gage = dfmdf[dfmdf['QAGE_3G'] == g].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_3gage =  pd.merge(hdfplaces_3gage, mdfplaces_3gage, on=['IncPlaceGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = places_3gage.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
for s in sexcats:
    hdfplaces_sex = dfhdf[dfhdf['QSEX'] == s].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_sex = dfmdf[dfmdf['QSEX'] == s].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_sex =  pd.merge(hdfplaces_sex, mdfplaces_sex, on=['IncPlaceGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = places_sex.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
for s in sexcats:
    for g in qage_3g_cats:
        hdfplaces_sex3gage = dfhdf[(dfhdf['QAGE_3G'] == g) & (dfhdf['QSEX']== s)].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
        mdfplaces_sex3gage = dfmdf[(dfmdf['QAGE_3G'] == g) & (dfmdf['QSEX']== s)].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
        places_sex3gage =  pd.merge(hdfplaces_sex3gage, mdfplaces_sex3gage, on=['IncPlaceGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = places_sex3gage.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

# Place Sex by 3 Age Groups [0.0, 500.0)
# Must use .reindex(places_lt500index, fill_value=0).reset_index()
for g in qage_3g_cats:
    hdfplaces_3gage = dfhdf[dfhdf['QAGE_3G'] == g].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_3gage = dfmdf[dfmdf['QAGE_3G'] == g].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_3gage =  pd.merge(hdfplaces_3gage, mdfplaces_3gage, on=['IncPlaceGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = places_3gage.pipe(calculate_ss, geography="Place", sizecategory = "[0.0, 500.0)", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
for s in sexcats:
    hdfplaces_sex = dfhdf[dfhdf['QSEX'] == s].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_sex = dfmdf[dfmdf['QSEX'] == s].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_sex =  pd.merge(hdfplaces_sex, mdfplaces_sex, on=['IncPlaceGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = places_sex.pipe(calculate_ss, geography="Place", sizecategory = "[0.0, 500.0)", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
for s in sexcats:
    for g in qage_3g_cats:
        hdfplaces_sex3gage = dfhdf[(dfhdf['QAGE_3G'] == g) & (dfhdf['QSEX']== s)].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
        mdfplaces_sex3gage = dfmdf[(dfmdf['QAGE_3G'] == g) & (dfmdf['QSEX']== s)].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
        places_sex3gage =  pd.merge(hdfplaces_sex3gage, mdfplaces_sex3gage, on=['IncPlaceGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = places_sex3gage.pipe(calculate_ss, geography="Place", sizecategory = "[0.0, 500.0)", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

# Tract Sex by 3 Age Groups
for g in qage_3g_cats:
    hdftracts_3gage = dfhdf[dfhdf['QAGE_3G'] == g].group_by(['TractGEOID']).agg(HDF_Population = pl.len())
    mdftracts_3gage = dfmdf[dfmdf['QAGE_3G'] == g].group_by(['TractGEOID']).agg(MDF_Population = pl.len())
    tracts_3gage =  pd.merge(hdftracts_3gage, mdftracts_3gage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = tracts_3gage.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
for s in sexcats:
    hdftracts_sex = dfhdf[(dfhdf['QSEX']== s)].group_by(['TractGEOID']).agg(HDF_Population = pl.len())
    mdftracts_sex = dfmdf[(dfmdf['QSEX']== s)].group_by(['TractGEOID']).agg(MDF_Population = pl.len())
    tracts_sex =  pd.merge(hdftracts_sex, mdftracts_sex, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = tracts_sex.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
for s in sexcats:
    for g in qage_3g_cats:
        hdftracts_sex3gage = dfhdf[(dfhdf['QAGE_3G'] == g) & (dfhdf['QSEX']== s)].group_by(['TractGEOID']).agg(HDF_Population = pl.len())
        mdftracts_sex3gage = dfmdf[(dfmdf['QAGE_3G'] == g) & (dfmdf['QSEX']== s)].group_by(['TractGEOID']).agg(MDF_Population = pl.len())
        tracts_sex3gage =  pd.merge(hdftracts_sex3gage, mdftracts_sex3gage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = tracts_sex3gage.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)
print("{} Sex By 3 Age Groups Done".format(datetime.now()))

# County 5-Year Age Groups
for g in qage_5y_cats:
    hdfcounties_5yage = dfhdf[dfhdf['QAGE_5Y'] == g].group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_5yage = dfmdf[dfmdf['QAGE_5Y'] == g].group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_5yage =  pd.merge(hdfcounties_5yage, mdfcounties_5yage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = counties_5yage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
# County Sex x 5-Year Age Groups
for s in sexcats:
    for g in qage_5y_cats:
        hdfcounties_sex5yage = dfhdf[(dfhdf['QAGE_5Y'] == g) & (dfhdf['QSEX']== s)].group_by('CountyGEOID').agg(HDF_Population = pl.len())
        mdfcounties_sex5yage = dfmdf[(dfmdf['QAGE_5Y'] == g) & (dfmdf['QSEX']== s)].group_by('CountyGEOID').agg(MDF_Population = pl.len())
        counties_sex5yage =  pd.merge(hdfcounties_sex5yage, mdfcounties_sex5yage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = counties_sex5yage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

if runAGEBYRACE:
    # County 5-Year Age Groups by RACEALONE
    for r in racealonecats:
        # for g in qage_5y_cats:
        #     hdfcounties_5yage = dfhdf[(dfhdf['QAGE_5Y'] == g)&(dfhdf['RACEALONE'] == r)].group_by('CountyGEOID').agg(HDF_Population = pl.len())
        #     mdfcounties_5yage = dfmdf[(dfmdf['QAGE_5Y'] == g)&(dfmdf['RACEALONE'] == r)].group_by('CountyGEOID').agg(MDF_Population = pl.len())
        #     counties_5yage =  pd.merge(hdfcounties_5yage, mdfcounties_5yage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        #     ss = counties_5yage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{race} Age {agegroup}".format(race = racealonedict.get(r),agegroup=g))
        #     outputdflist.append(ss)
        for s in sexcats:
            for g in qage_5y_cats:
                hdfcounties_sex5yage = dfhdf[(dfhdf['QAGE_5Y'] == g) & (dfhdf['QSEX']== s)&(dfhdf['RACEALONE'] == r)].group_by('CountyGEOID').agg(HDF_Population = pl.len())
                mdfcounties_sex5yage = dfmdf[(dfmdf['QAGE_5Y'] == g) & (dfmdf['QSEX']== s)&(dfmdf['RACEALONE'] == r)].group_by('CountyGEOID').agg(MDF_Population = pl.len())
                counties_sex5yage =  pd.merge(hdfcounties_sex5yage, mdfcounties_sex5yage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
                ss = counties_sex5yage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{race} {sex} Age {agegroup}".format(race = racealonedict.get(r), sex = sexdict.get(s), agegroup = g))
                outputdflist.append(ss)

    # County 5-Year Age Groups by RACE AOIC
    for rg in racegroups:
        # for g in qage_5y_cats:
        #     hdfcounties_5yage = dfhdf[(dfhdf['QAGE_5Y'] == g)&(dfhdf[rg]==1)].group_by('CountyGEOID').agg(HDF_Population = pl.len())
        #     mdfcounties_5yage = dfmdf[(dfmdf['QAGE_5Y'] == g)&(dfmdf[rg]==1)].group_by('CountyGEOID').agg(MDF_Population = pl.len())
        #     counties_5yage =  pd.merge(hdfcounties_5yage, mdfcounties_5yage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        #     ss = counties_5yage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{race} Age {agegroup}".format(race = raceincombdict.get(rg),agegroup=g))
        #     outputdflist.append(ss)
        for s in sexcats:
            for g in qage_5y_cats:
                hdfcounties_sex5yage = dfhdf[(dfhdf['QAGE_5Y'] == g) & (dfhdf['QSEX']== s)&(dfhdf[rg]==1)].group_by('CountyGEOID').agg(HDF_Population = pl.len())
                mdfcounties_sex5yage = dfmdf[(dfmdf['QAGE_5Y'] == g) & (dfmdf['QSEX']== s)&(dfmdf[rg]==1)].group_by('CountyGEOID').agg(MDF_Population = pl.len())
                counties_sex5yage =  pd.merge(hdfcounties_sex5yage, mdfcounties_sex5yage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
                ss = counties_sex5yage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{race} {sex} Age {agegroup}".format(race = raceincombdict.get(rg), sex = sexdict.get(s), agegroup = g))
                outputdflist.append(ss)

# Place 5-Year Age Groups
for g in qage_5y_cats:
    hdfplaces_5yage = dfhdf[dfhdf['QAGE_5Y'] == g].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_5yage = dfmdf[dfmdf['QAGE_5Y'] == g].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_5yage =  pd.merge(hdfplaces_5yage, mdfplaces_5yage, on=['IncPlaceGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = places_5yage.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
# Place Sex x 5-Year Age Groups
for s in sexcats:
    for g in qage_5y_cats:
        hdfplaces_sex5yage = dfhdf[(dfhdf['QAGE_5Y'] == g) & (dfhdf['QSEX']== s)].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
        mdfplaces_sex5yage = dfmdf[(dfmdf['QAGE_5Y'] == g) & (dfmdf['QSEX']== s)].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
        places_sex5yage =  pd.merge(hdfplaces_sex5yage, mdfplaces_sex5yage, on=['IncPlaceGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = places_sex5yage.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)


# Tract 5-Year Age Groups
for g in qage_5y_cats:
    hdftracts_5yage = dfhdf[dfhdf['QAGE_5Y'] == g].group_by(['TractGEOID']).agg(HDF_Population = pl.len())
    mdftracts_5yage = dfmdf[dfmdf['QAGE_5Y'] == g].group_by(['TractGEOID']).agg(MDF_Population = pl.len())
    tracts_5yage =  pd.merge(hdftracts_5yage, mdftracts_5yage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = tracts_5yage.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
# Tract Sex x 5-Year Age Groups
for s in sexcats:
    for g in qage_5y_cats:
        hdftracts_sex5yage = dfhdf[(dfhdf['QAGE_5Y'] == g) & (dfhdf['QSEX']== s)].group_by(['TractGEOID']).agg(HDF_Population = pl.len())
        mdftracts_sex5yage = dfmdf[(dfmdf['QAGE_5Y'] == g) & (dfmdf['QSEX']== s)].group_by(['TractGEOID']).agg(MDF_Population = pl.len())
        tracts_sex5yage =  pd.merge(hdftracts_sex5yage, mdftracts_sex5yage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = tracts_sex5yage.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

if runPRhere:
    # PR Counties/Municipios Sex and 5-Year Age Groups
    for s in sexcats:
        hdfcountiespr_sex = dfhdfpr[(dfhdfpr['QSEX']== s)].group_by('CountyGEOID').agg(HDF_Population = pl.len())
        mdfcountiespr_sex = dfmdfpr[(dfmdfpr['QSEX']== s)].group_by('CountyGEOID').agg(MDF_Population = pl.len())
        countiespr_sex =  pd.merge(hdfcountiespr_sex, mdfcountiespr_sex, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = countiespr_sex.pipe(calculate_ss, geography="PR County/Municipio", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
        outputdflist.append(ss)
    # PR Counties/Municipios 5-Year Age Groups
    for g in qage_5y_cats:
        hdfcountiespr_5yage = dfhdfpr[dfhdfpr['QAGE_5Y'] == g].group_by('CountyGEOID').agg(HDF_Population = pl.len())
        mdfcountiespr_5yage = dfmdfpr[dfmdfpr['QAGE_5Y'] == g].group_by('CountyGEOID').agg(MDF_Population = pl.len())
        countiespr_5yage =  pd.merge(hdfcountiespr_5yage, mdfcountiespr_5yage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = countiespr_5yage.pipe(calculate_ss, geography="PR County/Municipio", sizecategory = "All", characteristic = "Age {}".format(g))
        outputdflist.append(ss)
    # PR Counties/Municipios Sex x 5-Year Age Groups
    for s in sexcats:
        for g in qage_5y_cats:
            hdfcountiespr_sex5yage = dfhdfpr[(dfhdfpr['QAGE_5Y'] == g) & (dfhdfpr['QSEX']== s)].group_by('CountyGEOID').agg(HDF_Population = pl.len())
            mdfcountiespr_sex5yage = dfmdfpr[(dfmdfpr['QAGE_5Y'] == g) & (dfmdfpr['QSEX']== s)].group_by('CountyGEOID').agg(MDF_Population = pl.len())
            countiespr_sex5yage =  pd.merge(hdfcountiespr_sex5yage, mdfcountiespr_sex5yage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
            ss = countiespr_sex5yage.pipe(calculate_ss, geography="PR County/Municipio", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
            outputdflist.append(ss)
    # PR Tract Sex and 5-Year Age Groups
    for s in sexcats:
        hdftractspr_sex = dfhdfpr[(dfhdfpr['QSEX']== s)].group_by(['TractGEOID']).agg(HDF_Population = pl.len())
        mdftractspr_sex = dfmdfpr[(dfmdfpr['QSEX']== s)].group_by(['TractGEOID']).agg(MDF_Population = pl.len())
        tractspr_sex =  pd.merge(hdftractspr_sex, mdftractspr_sex, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = tractspr_sex.pipe(calculate_ss, geography="PR Tract", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
        outputdflist.append(ss)
    # PR Tract 5-Year Age Groups
    for g in qage_5y_cats:
        hdftractspr_5yage = dfhdfpr[dfhdfpr['QAGE_5Y'] == g].group_by(['TractGEOID']).agg(HDF_Population = pl.len())
        mdftractspr_5yage = dfmdfpr[dfmdfpr['QAGE_5Y'] == g].group_by(['TractGEOID']).agg(MDF_Population = pl.len())
        tractspr_5yage =  pd.merge(hdftractspr_5yage, mdftractspr_5yage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = tractspr_5yage.pipe(calculate_ss, geography="PR Tract", sizecategory = "All", characteristic = "Age {}".format(g))
        outputdflist.append(ss)
    # PR Tract Sex x 5-Year Age Groups
    for s in sexcats:
        for g in qage_5y_cats:
            hdftractspr_sex5yage = dfhdfpr[(dfhdfpr['QAGE_5Y'] == g) & (dfhdfpr['QSEX']== s)].group_by(['TractGEOID']).agg(HDF_Population = pl.len())
            mdftractspr_sex5yage = dfmdfpr[(dfmdfpr['QAGE_5Y'] == g) & (dfmdfpr['QSEX']== s)].group_by(['TractGEOID']).agg(MDF_Population = pl.len())
            tractspr_sex5yage =  pd.merge(hdftractspr_sex5yage, mdftractspr_sex5yage, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
            ss = tractspr_sex5yage.pipe(calculate_ss, geography="PR Tract", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
            outputdflist.append(ss)

# Federal AIR Sex and 5-Year Age Groups
for s in sexcats:
    hdffedairs_sex = dfhdf[(dfhdf['QSEX']== s)].group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
    mdffedairs_sex = dfmdf[(dfmdf['QSEX']== s)].group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
    fedairs_sex =  pd.merge(hdffedairs_sex, mdffedairs_sex, on=['FedAIRGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = fedairs_sex.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
# Federal AIR 5-Year Age Groups
for g in qage_5y_cats:
    hdffedairs_5yage = dfhdf[dfhdf['QAGE_5Y'] == g].group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
    mdffedairs_5yage = dfmdf[dfmdf['QAGE_5Y'] == g].group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
    fedairs_5yage =  pd.merge(hdffedairs_5yage, mdffedairs_5yage, on=['FedAIRGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = fedairs_5yage.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
#    Federal AIR Sex x 5-Year Age Groups
for s in sexcats:
    for g in qage_5y_cats:
        hdffedairs_sex5yage = dfhdf[(dfhdf['QAGE_5Y'] == g) & (dfhdf['QSEX']== s)].group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
        mdffedairs_sex5yage = dfmdf[(dfmdf['QAGE_5Y'] == g) & (dfmdf['QSEX']== s)].group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
        fedairs_sex5yage =  pd.merge(hdffedairs_sex5yage, mdffedairs_sex5yage, on=['FedAIRGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = fedairs_sex5yage.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

if runAGEBYRACE:
    # Federal AIR 5-Year Age Groups by RACE ALONE
    for r in racealonecats:
        for s in sexcats:
            for g in qage_5y_cats:
                hdffedairs_sex5yage = dfhdf[(dfhdf['QAGE_5Y'] == g) & (dfhdf['QSEX']== s)&(dfhdf['RACEALONE'] == r)].group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
                mdffedairs_sex5yage = dfmdf[(dfmdf['QAGE_5Y'] == g) & (dfmdf['QSEX']== s)&(dfmdf['RACEALONE'] == r)].group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
                fedairs_sex5yage =  pd.merge(hdffedairs_sex5yage, mdffedairs_sex5yage, on=['FedAIRGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
                ss = fedairs_sex5yage.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "{race} {sex} Age {agegroup}".format(race = racealonedict.get(r), sex = sexdict.get(s), agegroup = g))
                outputdflist.append(ss)
    # Fed AIR 5-Year Age Groups by RACE AOIC
    for rg in racegroups:
        for s in sexcats:
            for g in qage_5y_cats:
                hdffedairs_sex5yage = dfhdf[(dfhdf['QAGE_5Y'] == g) & (dfhdf['QSEX']== s) & (dfhdf[rg]==1)].group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
                mdffedairs_sex5yage = dfmdf[(dfmdf['QAGE_5Y'] == g) & (dfmdf['QSEX']== s) & (dfmdf[rg]==1)].group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
                fedairs_sex5yage =  pd.merge(hdffedairs_sex5yage, mdffedairs_sex5yage, on=['FedAIRGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
                ss = fedairs_sex5yage.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "{race} {sex} Age {agegroup}".format(race = raceincombdict.get(rg), sex = sexdict.get(s), agegroup = g))
                outputdflist.append(ss)


# OTSA Sex and 5-Year Age Groups
for s in sexcats:
    hdfotsas_sex = dfhdf[(dfhdf['QSEX']== s)].group_by(['OTSAGEOID']).agg(HDF_Population = pl.len())
    mdfotsas_sex = dfmdf[(dfmdf['QSEX']== s)].group_by(['OTSAGEOID']).agg(MDF_Population = pl.len())
    otsas_sex =  pd.merge(hdfotsas_sex, mdfotsas_sex, on=['OTSAGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = otsas_sex.pipe(calculate_ss, geography="OTSA", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
# OTSA 5-Year Age Groups
for g in qage_5y_cats:
    hdfotsas_5yage = dfhdf[(dfhdf['QAGE_5Y'] == g)].group_by(['OTSAGEOID']).agg(HDF_Population = pl.len())
    mdfotsas_5yage = dfmdf[(dfmdf['QAGE_5Y'] == g)].group_by(['OTSAGEOID']).agg(MDF_Population = pl.len())
    otsas_5yage =  pd.merge(hdfotsas_5yage, mdfotsas_5yage, on=['OTSAGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = otsas_5yage.pipe(calculate_ss, geography="OTSA", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
# OTSA Sex x 5-Year Age Groups
for s in sexcats:
    for g in qage_5y_cats:
        hdfotsas_sex5yage = dfhdf[(dfhdf['QAGE_5Y'] == g) & (dfhdf['QSEX']== s)].group_by(['OTSAGEOID']).agg(HDF_Population = pl.len())
        mdfotsas_sex5yage = dfmdf[(dfmdf['QAGE_5Y'] == g) & (dfmdf['QSEX']== s)].group_by(['OTSAGEOID']).agg(MDF_Population = pl.len())
        otsas_sex5yage =  pd.merge(hdfotsas_sex5yage, mdfotsas_sex5yage, on=['OTSAGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = otsas_sex5yage.pipe(calculate_ss, geography="OTSA", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

# ANVSA Sex and 5-Year Age Groups
for s in sexcats:
    hdfanvsas_sex = dfhdf[(dfhdf['QSEX']== s)].group_by(['ANVSAGEOID']).agg(HDF_Population = pl.len())
    mdfanvsas_sex = dfmdf[(dfmdf['QSEX']== s)].group_by(['ANVSAGEOID']).agg(MDF_Population = pl.len())
    anvsas_sex =  pd.merge(hdfanvsas_sex, mdfanvsas_sex, on=['ANVSAGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = anvsas_sex.pipe(calculate_ss, geography="ANVSA", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
# ANVSA 5-Year Age Groups
for g in qage_5y_cats:
    hdfanvsas_5yage = dfhdf[dfhdf['QAGE_5Y'] == g].group_by(['ANVSAGEOID']).agg(HDF_Population = pl.len())
    mdfanvsas_5yage = dfmdf[dfmdf['QAGE_5Y'] == g].group_by(['ANVSAGEOID']).agg(MDF_Population = pl.len())
    anvsas_5yage =  pd.merge(hdfanvsas_5yage, mdfanvsas_5yage, on=['ANVSAGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = anvsas_5yage.pipe(calculate_ss, geography="ANVSA", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
# ANVSA Sex x 5-Year Age Groups
for s in sexcats:
    for g in qage_5y_cats:
        hdfanvsas_sex5yage = dfhdf[(dfhdf['QAGE_5Y'] == g) & (dfhdf['QSEX']== s)].group_by(['ANVSAGEOID']).agg(HDF_Population = pl.len())
        mdfanvsas_sex5yage = dfmdf[(dfmdf['QAGE_5Y'] == g) & (dfmdf['QSEX']== s)].group_by(['ANVSAGEOID']).agg(MDF_Population = pl.len())
        anvsas_sex5yage =  pd.merge(hdfanvsas_sex5yage, mdfanvsas_sex5yage, on=['ANVSAGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
        ss = anvsas_sex5yage.pipe(calculate_ss, geography="ANVSA", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

print("{} Sex By 5 Year Age Groups Done".format(datetime.now()))

# GQ 
for g in gqinstcats:
    hdfstates_gqinst = dfhdf[dfhdf['GQINST'] == g].group_by(['TABBLKST']).agg(HDF_Population = pl.len())
    mdfstates_gqinst = dfmdf[dfmdf['GQINST'] == g].group_by(['TABBLKST']).agg(MDF_Population = pl.len())
    states_gqinst =  pd.merge(hdfstates_gqinst, mdfstates_gqinst, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = states_gqinst.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)

for g in gqmajortypecats:
    hdfstates_gqmajtype = dfhdf[dfhdf['GQMAJORTYPE'] == g].group_by(['TABBLKST']).agg(HDF_Population = pl.len())
    mdfstates_gqmajtype = dfmdf[dfmdf['GQMAJORTYPE'] == g].group_by(['TABBLKST']).agg(MDF_Population = pl.len())
    states_gqmajtype =  pd.merge(hdfstates_gqmajtype, mdfstates_gqmajtype, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = states_gqmajtype.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)

for g in gqinstcats:
    hdfcounties_gqinst = dfhdf[dfhdf['GQINST'] == g].group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_gqinst = dfmdf[dfmdf['GQINST'] == g].group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_gqinst =  pd.merge(hdfcounties_gqinst, mdfcounties_gqinst, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = counties_gqinst.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)
    counties_gqinst = pd.merge(counties_gqinst, dfhdf().group_by('CountyGEOID').size().reset_index(name='HDF_TotalPopulation').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']), on ="GEOID", how="outer",validate=mergeValidation)
    counties_gqinst['Total_PopSize'] = pd.cut(counties_gqinst['HDF_TotalPopulation'], [0,1000,5000,10000,50000,100000,np.inf], left_closed = True)
    for i in counties_gqinst['Total_PopSize'].cat.get_categories():
        ss = counties_gqinst[counties_gqinst['Total_PopSize'] == i].pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "GQ {}".format(g))
        outputdflist.append(ss)

for g in gqmajortypecats:
    hdfcounties_gqmajtype = dfhdf[dfhdf['GQMAJORTYPE'] == g].group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_gqmajtype = dfmdf[dfmdf['GQMAJORTYPE'] == g].group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_gqmajtype =  pd.merge(hdfcounties_gqmajtype, mdfcounties_gqmajtype, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = counties_gqmajtype.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)
    counties_gqmajtype = pd.merge(counties_gqmajtype, dfhdf().group_by('CountyGEOID').size().reset_index(name='HDF_TotalPopulation').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']), on ="GEOID", how="outer",validate=mergeValidation)
    counties_gqmajtype['Total_PopSize'] = pd.cut(counties_gqmajtype['HDF_TotalPopulation'], [0,1000,5000,10000,50000,100000,np.inf], left_closed = True)
    for i in counties_gqmajtype['Total_PopSize'].cat.get_categories():
        ss = counties_gqmajtype[counties_gqmajtype['Total_PopSize'] == i].pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "GQ {}".format(g))
        outputdflist.append(ss)

for g in gqinstcats:
    hdfplaces_gqinst = dfhdf[dfhdf['GQINST'] == g].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_gqinst = dfmdf[dfmdf['GQINST'] == g].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_gqinst =  pd.merge(hdfplaces_gqinst, mdfplaces_gqinst, on=['IncPlaceGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = places_gqinst.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)
    places_gqinst = pd.merge(places_gqinst, dfhdf().group_by(['IncPlaceGEOID']).size().reset_index(name='HDF_TotalPopulation').set_index(['IncPlaceGEOID']).reindex(allplacesindex, fill_value=0).reset_index(), on ="IncPlaceGEOID", how="outer",validate=mergeValidation)
    places_gqinst['Total_PopSize'] = pd.cut(places_gqinst['HDF_TotalPopulation'], [0,500,1000,5000,10000,50000,100000,np.inf], left_closed = True)
    for i in places_gqinst['Total_PopSize'].cat.get_categories():
        ss = places_gqinst[places_gqinst['Total_PopSize'] == i].pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "GQ {}".format(g))
        outputdflist.append(ss)

for g in gqmajortypecats:
    hdfplaces_gqmajtype = dfhdf[dfhdf['GQMAJORTYPE'] == g].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_gqmajtype = dfmdf[dfmdf['GQMAJORTYPE'] == g].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_gqmajtype =  pd.merge(hdfplaces_gqmajtype, mdfplaces_gqmajtype, on=['IncPlaceGEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = places_gqmajtype.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)
    places_gqmajtype = pd.merge(places_gqmajtype, dfhdf().group_by(['IncPlaceGEOID']).size().reset_index(name='HDF_TotalPopulation').set_index(['IncPlaceGEOID']).reindex(allplacesindex, fill_value=0).reset_index(), on ="IncPlaceGEOID", how="outer",validate=mergeValidation)
    places_gqmajtype['Total_PopSize'] = pd.cut(places_gqmajtype['HDF_TotalPopulation'], [0,500,1000,5000,10000,50000,100000,np.inf], left_closed = True)
    for i in places_gqmajtype['Total_PopSize'].cat.get_categories():
        ss = places_gqmajtype[places_gqmajtype['Total_PopSize'] == i].pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "GQ {}".format(g))
        outputdflist.append(ss)

for g in gqinstcats:
    hdftracts_gqinst = dfhdf[dfhdf['GQINST'] == g].group_by(['TractGEOID']).agg(HDF_Population = pl.len())
    mdftracts_gqinst = dfmdf[dfmdf['GQINST'] == g].group_by(['TractGEOID']).agg(MDF_Population = pl.len())
    tracts_gqinst =  pd.merge(hdftracts_gqinst, mdftracts_gqinst, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = tracts_gqinst.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)

for g in gqmajortypecats:
    hdftracts_gqmajtype = dfhdf[dfhdf['GQMAJORTYPE'] == g].group_by(['TractGEOID']).agg(HDF_Population = pl.len())
    mdftracts_gqmajtype = dfmdf[dfmdf['GQMAJORTYPE'] == g].group_by(['TractGEOID']).agg(MDF_Population = pl.len())
    tracts_gqmajtype =  pd.merge(hdftracts_gqmajtype, mdftracts_gqmajtype, on=['GEOID'], how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = tracts_gqmajtype.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)

print("{} GQ Types Done".format(datetime.now()))



# Counties Absolute Change in Median Age/Sex Ratio 
hdfcounties_medage = dfhdf().group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_MedianAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index(['GEOID']).reindex(allcountiesindex, fill_value=0).reset_index()
mdfcounties_medage = dfmdf().group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_MedianAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index(['GEOID']).reindex(allcountiesindex, fill_value=0).reset_index()
counties_medage =  pd.merge(hdfcounties_medage, mdfcounties_medage, on='GEOID', how = 'outer', validate = mergeValidation)
counties_medage  = counties_medage.assign(AbsDiffMedAge = lambda x: np.abs(x['HDF_MedianAge'] - x['MDF_MedianAge']))
counties_medage.to_csv(f"{OUTPUTDIR}/counties_medage.csv", index=False)
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'Average Absolute Change in Median Age','NumCells':len(counties_medage),'AvgAbsDiffMedAge': np.nanmean(counties_medage['AbsDiffMedAge'])}, index=[0])
outputdflist.append(ss)
counties_medage = pd.merge(counties_medage, dfhdf().group_by('CountyGEOID').size().reset_index(name='HDF_TotalPopulation').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']), on ="GEOID", how="outer",validate=mergeValidation)
counties_medage['Total_PopSize'] = pd.cut(counties_medage['HDF_TotalPopulation'], [0,1000,5000,10000,50000,100000,np.inf], left_closed = True)
for i in counties_medage['Total_PopSize'].cat.get_categories():
    ss = pd.DataFrame({'Geography':'County', 'Size_Category':str(i), 'Characteristic':'Average Absolute Change in Median Age','NumCells':len(counties_medage[counties_medage['Total_PopSize'] == i]),'AvgAbsDiffMedAge': np.nanmean(counties_medage.loc[counties_medage['Total_PopSize'] == i,'AbsDiffMedAge'])}, index=[0])
    outputdflist.append(ss)

hdfcounties_sexratio = dfhdf().group_by(['TABBLKST', 'TABBLKCOU', 'QSEX']).size().unstack().fillna(0).reset_index().assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index(['GEOID']).reindex(allcountiesindex, fill_value=0).reset_index().assign(HDF_SexRatio = lambda x: 100*x['1']/x['2']).drop(columns=['1','2'])
mdfcounties_sexratio = dfmdf().group_by(['TABBLKST', 'TABBLKCOU', 'QSEX']).size().unstack().fillna(0).reset_index().assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index(['GEOID']).reindex(allcountiesindex, fill_value=0).reset_index().assign(MDF_SexRatio = lambda x: 100*x['1']/x['2']).drop(columns=['1','2'])
counties_sexratio =  pd.merge(hdfcounties_sexratio, mdfcounties_sexratio, on='GEOID', how = 'outer', validate = mergeValidation)
counties_sexratio  = counties_sexratio.assign(AbsDiffSexRatio = lambda x: np.abs(x['HDF_SexRatio'] - x['MDF_SexRatio']))
counties_sexratio.to_csv(f"{OUTPUTDIR}/counties_sexratio.csv", index=False)
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'Average Absolute Change in Sex Ratio','NumCells':len(counties_sexratio),'AvgAbsDiffSexRatio': np.nanmean(counties_sexratio['AbsDiffSexRatio'])}, index=[0])
outputdflist.append(ss)
counties_sexratio = pd.merge(counties_sexratio, dfhdf().group_by('CountyGEOID').size().reset_index(name='HDF_TotalPopulation').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']), on ="GEOID", how="outer",validate=mergeValidation)
counties_sexratio['Total_PopSize'] = pd.cut(counties_sexratio['HDF_TotalPopulation'], [0,1000,5000,10000,50000,100000,np.inf], left_closed = True)
for i in counties_sexratio['Total_PopSize'].cat.get_categories():
    ss = pd.DataFrame({'Geography':'County', 'Size_Category':str(i), 'Characteristic':'Average Absolute Change in Sex Ratio','NumCells':len(counties_sexratio[counties_sexratio['Total_PopSize'] == i]),'AvgAbsDiffSexRatio': np.nanmean(counties_sexratio.loc[counties_sexratio['Total_PopSize'] == i,'AbsDiffSexRatio'])}, index=[0])
    outputdflist.append(ss)


# Counties GQ Absolute Change in Median Age/Sex Ratio 
hdfcountiesgq_medage = dfhdf[dfhdf['GQTYPE'] > 0].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_MedianAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index(['GEOID']).reindex(allcountiesindex, fill_value=0).reset_index()
mdfcountiesgq_medage = dfmdf[dfmdf['GQTYPE'] > 0].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_MedianAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index(['GEOID']).reindex(allcountiesindex, fill_value=0).reset_index()
countiesgq_medage =  pd.merge(hdfcountiesgq_medage, mdfcountiesgq_medage, on='GEOID', how = 'outer', validate = mergeValidation)
countiesgq_medage  = countiesgq_medage.assign(AbsDiffMedAge = lambda x: np.abs(x['HDF_MedianAge'] - x['MDF_MedianAge']))
countiesgq_medage.to_csv(f"{OUTPUTDIR}/countiesgq_medage.csv", index=False)
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'Average Absolute Change in Median Age of GQ Population','NumCells':len(countiesgq_medage),'AvgAbsDiffMedAge': np.nanmean(countiesgq_medage['AbsDiffMedAge'])}, index=[0])
outputdflist.append(ss)
countiesgq_medage = pd.merge(countiesgq_medage, dfhdf().group_by('CountyGEOID').size().reset_index(name='HDF_TotalPopulation').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']), on ="GEOID", how="outer",validate=mergeValidation)
countiesgq_medage['Total_PopSize'] = pd.cut(countiesgq_medage['HDF_TotalPopulation'], [0,1000,5000,10000,50000,100000,np.inf], left_closed = True)
for i in countiesgq_medage['Total_PopSize'].cat.get_categories():
    ss = pd.DataFrame({'Geography':'County', 'Size_Category':str(i), 'Characteristic':'Average Absolute Change in Median Age of GQ Population','NumCells':len(countiesgq_medage[countiesgq_medage['Total_PopSize'] == i]),'AvgAbsDiffMedAge': np.nanmean(countiesgq_medage.loc[countiesgq_medage['Total_PopSize'] == i,'AbsDiffMedAge'])}, index=[0])
    outputdflist.append(ss)

hdfcountiesgq_sexratio = dfhdf[dfhdf['GQTYPE'] > 0].group_by(['TABBLKST', 'TABBLKCOU', 'QSEX']).size().unstack().fillna(0).reset_index().assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index(['GEOID']).reindex(allcountiesindex, fill_value=0).reset_index().assign(HDF_SexRatio = lambda x: 100*x['1']/x['2']).drop(columns=['1','2'])
mdfcountiesgq_sexratio = dfmdf[dfmdf['GQTYPE'] > 0].group_by(['TABBLKST', 'TABBLKCOU', 'QSEX']).size().unstack().fillna(0).reset_index().assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index(['GEOID']).reindex(allcountiesindex, fill_value=0).reset_index().assign(MDF_SexRatio = lambda x: 100*x['1']/x['2']).drop(columns=['1','2'])
countiesgq_sexratio =  pd.merge(hdfcountiesgq_sexratio, mdfcountiesgq_sexratio, on='GEOID', how = 'outer', validate = mergeValidation)
countiesgq_sexratio  = countiesgq_sexratio.assign(AbsDiffSexRatio = lambda x: np.abs(x['HDF_SexRatio'] - x['MDF_SexRatio']))
countiesgq_sexratio.to_csv(f"{OUTPUTDIR}/countiesgq_sexratio.csv", index=False)
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'Average Absolute Change in Sex Ratio of GQ Population','NumCells':len(countiesgq_sexratio),'AvgAbsDiffSexRatio': np.nanmean(countiesgq_sexratio['AbsDiffSexRatio'])}, index=[0])
outputdflist.append(ss)
countiesgq_sexratio = pd.merge(countiesgq_sexratio, dfhdf().group_by('CountyGEOID').size().reset_index(name='HDF_TotalPopulation').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']), on ="GEOID", how="outer",validate=mergeValidation)
countiesgq_sexratio['Total_PopSize'] = pd.cut(countiesgq_sexratio['HDF_TotalPopulation'], [0,1000,5000,10000,50000,100000,np.inf], left_closed = True)
for i in countiesgq_sexratio['Total_PopSize'].cat.get_categories():
    ss = pd.DataFrame({'Geography':'County', 'Size_Category':str(i), 'Characteristic':'Average Absolute Change in Sex Ratio of GQ Population','NumCells':len(countiesgq_sexratio[countiesgq_sexratio['Total_PopSize'] == i]),'AvgAbsDiffSexRatio': np.nanmean(countiesgq_sexratio.loc[countiesgq_sexratio['Total_PopSize'] == i,'AbsDiffSexRatio'])}, index=[0])
    outputdflist.append(ss)

print("{} Average Absolute Change in Median Age and Sex Ratio Done".format(datetime.now()))


print("{} Starting Use Cases".format(datetime.now()))

# Tracts Aged 75+
hdftracts_over75 = dfhdf[dfhdf['QAGE']>= 75].group_by('TractGEOID').agg(HDF_Population = pl.len())
mdftracts_over75 = dfmdf[dfmdf['QAGE']>= 75].group_by('TractGEOID').agg(MDF_Population = pl.len())
tracts_over75 =  pd.merge(hdftracts_over75, mdftracts_over75, on='GEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
ss = tracts_over75.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Aged 75 and Over")
outputdflist.append(ss)

# Counties and Places By State TAES
for s in allstates:
    dfhdfstate = dfhdf[dfhdf['TABBLKST'] == s]
    dfmdfstate = dfmdf[dfmdf['TABBLKST'] == s]
    hdfcounties_taes = dfhdfstate.group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_taes = dfmdfstate.group_by('CountyGEOID').agg(MDF_Population = pl.len())
    hdfsize = len(dfhdfstate)
    mdfsize = len(dfmdfstate)
    counties_taes =  pd.merge(hdfcounties_taes, mdfcounties_taes, on='GEOID', how = 'outer', validate = mergeValidation)
    counties_taes = counties_taes.fillna({'HDF_Population': 0, 'MDF_Population': 0})
    counties_taes  = counties_taes.assign(AES = lambda x: np.abs(x['HDF_Population']/hdfsize - x['MDF_Population']/mdfsize))
    ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'TAES Total Population {}'.format(statedict.get(s)),'NumCells':len(counties_taes),'TAES': np.sum(counties_taes['AES'])}, index=[0])
    outputdflist.append(ss)
    if s == "15":
        ss = pd.DataFrame({'Geography':'Place', 'Size_Category':'All', 'Characteristic':'TAES Total Population {}'.format(statedict.get(s)),'NumCells':0,'TAES': 0}, index=[0])
        outputdflist.append(ss)
    else:
        hdfplaces_taes = dfhdfstate.group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
        mdfplaces_taes = dfmdfstate.group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
        places_taes =  pd.merge(hdfplaces_taes, mdfplaces_taes, on='IncPlaceGEOID', how = 'outer', validate = mergeValidation)
        places_taes = places_taes.fillna({'HDF_Population': 0, 'MDF_Population': 0})
        places_taes  = places_taes.assign(AES = lambda x: np.abs(x['HDF_Population']/hdfsize - x['MDF_Population']/mdfsize))
        ss = pd.DataFrame({'Geography':'Place', 'Size_Category':'All', 'Characteristic':'TAES Total Population {}'.format(statedict.get(s)),'NumCells':len(places_taes),'TAES': np.sum(places_taes['AES'])}, index=[0])
        outputdflist.append(ss)

# MCDs By State TAES
for s in mcdstates:
    dfhdfstate = dfhdf[dfhdf['TABBLKST'] == s]
    dfmdfstate = dfmdf[dfmdf['TABBLKST'] == s]
    hdfmcds_taes = dfhdfstate.group_by(['MCDGEOID']).agg(HDF_Population = pl.len())
    mdfmcds_taes = dfmdfstate.group_by(['MCDGEOID']).agg(MDF_Population = pl.len())
    hdfsize = len(dfhdfstate)
    mdfsize = len(dfmdfstate)
    mcds_taes =  pd.merge(hdfmcds_taes, mdfmcds_taes, on='MCDGEOID', how = 'outer', validate = mergeValidation)
    mcds_taes = mcds_taes.fillna({'HDF_Population': 0, 'MDF_Population': 0})
    mcds_taes  = mcds_taes.assign(AES = lambda x: np.abs(x['HDF_Population']/hdfsize - x['MDF_Population']/mdfsize))
    ss = pd.DataFrame({'Geography':'MCD', 'Size_Category':'All', 'Characteristic':'TAES Total Population {}'.format(statedict.get(s)),'NumCells':len(mcds_taes),'TAES': np.sum(mcds_taes['AES'])}, index=[0])
    outputdflist.append(ss)

# Counties Single Year of Age < 18  
for y in list(range(0,18)):
    hdfcounties_age = dfhdf[dfhdf['QAGE'] == y].group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_age = dfmdf[dfmdf['QAGE'] == y].group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_age =  pd.merge(hdfcounties_age, mdfcounties_age, on='GEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = counties_age.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Age {}".format(y))
    outputdflist.append(ss)
    counties_age = pd.merge(counties_age, dfhdf[dfhdf['QAGE'] < 18].group_by('CountyGEOID').size().reset_index(name='HDF_PopulationUnder18').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']), on="GEOID", how="outer", validate=mergeValidation)
    counties_age['Under18_PopSize'] = pd.cut(counties_age['HDF_PopulationUnder18'], [0,1000,10000,np.inf], left_closed = True)
    for i in counties_age['Under18_PopSize'].cat.get_categories():
        ss = counties_age[counties_age['Under18_PopSize'] == i].pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Age {}".format(y))
        outputdflist.append(ss)

# Elem School Districts Single Year of Age < 18  
for y in list(range(0,18)):
    hdfelemschdists_age = dfhdf[dfhdf['QAGE'] == y].group_by(['SchDistEGEOID']).agg(HDF_Population = pl.len())
    mdfelemschdists_age = dfmdf[dfmdf['QAGE'] == y].group_by(['SchDistEGEOID']).agg(MDF_Population = pl.len())
    elemschdists_age =  pd.merge(hdfelemschdists_age, mdfelemschdists_age, on='SchDistEGEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = elemschdists_age.pipe(calculate_ss, geography="ESD", sizecategory = "All", characteristic = "Age {}".format(y))
    outputdflist.append(ss)
    elemschdists_age = pd.merge(elemschdists_age, dfhdf[dfhdf['QAGE'] < 18].group_by(['SchDistEGEOID']).size().reset_index(name='HDF_PopulationUnder18').set_index('SchDistEGEOID').reindex(allelemschdistsindex, fill_value=0).reset_index(), on="SchDistEGEOID", how="outer", validate=mergeValidation)
    elemschdists_age['Under18_PopSize'] = pd.cut(elemschdists_age['HDF_PopulationUnder18'], [0,1000,10000,np.inf], left_closed = True)
    for i in elemschdists_age['Under18_PopSize'].cat.get_categories():
        ss = elemschdists_age[elemschdists_age['Under18_PopSize'] == i].pipe(calculate_ss, geography="ESD", sizecategory = str(i), characteristic = "Age {}".format(y))
        outputdflist.append(ss)

# Sec School Districts Single Year of Age < 18  
for y in list(range(0,18)):
    hdfsecschdists_age = dfhdf[dfhdf['QAGE'] == y].group_by(['SchDistSGEOID']).agg(HDF_Population = pl.len())
    mdfsecschdists_age = dfmdf[dfmdf['QAGE'] == y].group_by(['SchDistSGEOID']).agg(MDF_Population = pl.len())
    secschdists_age =  pd.merge(hdfsecschdists_age, mdfsecschdists_age, on='SchDistSGEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = secschdists_age.pipe(calculate_ss, geography="SSD", sizecategory = "All", characteristic = "Age {}".format(y))
    outputdflist.append(ss)
    secschdists_age = pd.merge(secschdists_age, dfhdf[dfhdf['QAGE'] < 18].group_by(['SchDistSGEOID']).size().reset_index(name='HDF_PopulationUnder18').set_index('SchDistSGEOID').reindex(allsecschdistsindex, fill_value=0).reset_index(), on="SchDistSGEOID", how="outer", validate=mergeValidation)
    secschdists_age['Under18_PopSize'] = pd.cut(secschdists_age['HDF_PopulationUnder18'], [0,1000,10000,np.inf], left_closed = True)
    for i in secschdists_age['Under18_PopSize'].cat.get_categories():
        ss = secschdists_age[secschdists_age['Under18_PopSize'] == i].pipe(calculate_ss, geography="SSD", sizecategory = str(i), characteristic = "Age {}".format(y))
        outputdflist.append(ss)

# Uni School Districts Single Year of Age < 18  
for y in list(range(0,18)):
    hdfunischdists_age = dfhdf[dfhdf['QAGE'] == y].group_by(['SchDistUGEOID']).agg(HDF_Population = pl.len())
    mdfunischdists_age = dfmdf[dfmdf['QAGE'] == y].group_by(['SchDistUGEOID']).agg(MDF_Population = pl.len())
    unischdists_age =  pd.merge(hdfunischdists_age, mdfunischdists_age, on='SchDistUGEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
    ss = unischdists_age.pipe(calculate_ss, geography="USD", sizecategory = "All", characteristic = "Age {}".format(y))
    outputdflist.append(ss)
    unischdists_age = pd.merge(unischdists_age, dfhdf[dfhdf['QAGE'] < 18].group_by(['SchDistUGEOID']).size().reset_index(name='HDF_PopulationUnder18').set_index('SchDistUGEOID').reindex(allunischdistsindex, fill_value=0).reset_index(), on="SchDistUGEOID", how="outer", validate=mergeValidation)
    unischdists_age['Under18_PopSize'] = pd.cut(unischdists_age['HDF_PopulationUnder18'], [0,1000,10000,np.inf], left_closed = True)
    for i in unischdists_age['Under18_PopSize'].cat.get_categories():
        ss = unischdists_age[unischdists_age['Under18_PopSize'] == i].pipe(calculate_ss, geography="USD", sizecategory = str(i), characteristic = "Age {}".format(y))
        outputdflist.append(ss)

# Counties Nationwide AIAN Alone or In Combination TAES
hdfcounties_aianaloneorincomb_taes = dfhdf[dfhdf['aianalone-or-incomb'] == 1].group_by('CountyGEOID').agg(HDF_Population = pl.len())
mdfcounties_aianaloneorincomb_taes = dfmdf[dfmdf['aianalone-or-incomb'] == 1].group_by('CountyGEOID').agg(MDF_Population = pl.len())
hdfsize_aianaloneorincomb = len(dfhdf[dfhdf['aianalone-or-incomb'] == 1])
mdfsize_aianaloneorincomb = len(dfmdf[dfmdf['aianalone-or-incomb'] == 1])
counties_aianaloneorincomb_taes =  pd.merge(hdfcounties_aianaloneorincomb_taes, mdfcounties_aianaloneorincomb_taes, on='GEOID', how = 'outer', validate = mergeValidation)
counties_aianaloneorincomb_taes  = counties_aianaloneorincomb_taes.assign(AES = lambda x: np.abs(x['HDF_Population']/hdfsize_aianaloneorincomb - x['MDF_Population']/mdfsize_aianaloneorincomb))
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'TAES AIAN Alone or In Combination Nation','NumCells':len(counties_aianaloneorincomb_taes),'TAES': np.sum(counties_aianaloneorincomb_taes['AES'])}, index=[0])
outputdflist.append(ss)

# Places Nationwide AIAN Alone or In Combination TAES
hdfplaces_aianaloneorincomb_taes = dfhdf[dfhdf['aianalone-or-incomb'] == 1].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
mdfplaces_aianaloneorincomb_taes = dfmdf[dfmdf['aianalone-or-incomb'] == 1].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
# hdfsize_aianaloneorincomb = len(dfhdf[dfhdf['aianalone-or-incomb'] == 1])
# mdfsize_aianaloneorincomb = len(dfmdf[dfmdf['aianalone-or-incomb'] == 1])
places_aianaloneorincomb_taes =  pd.merge(hdfplaces_aianaloneorincomb_taes, mdfplaces_aianaloneorincomb_taes, on='IncPlaceGEOID', how = 'outer', validate = mergeValidation)
places_aianaloneorincomb_taes  = places_aianaloneorincomb_taes.assign(AES = lambda x: np.abs(x['HDF_Population']/hdfsize_aianaloneorincomb - x['MDF_Population']/mdfsize_aianaloneorincomb))
ss = pd.DataFrame({'Geography':'Place', 'Size_Category':'All', 'Characteristic':'TAES AIAN Alone or In Combination Nation','NumCells':len(places_aianaloneorincomb_taes),'TAES': np.sum(places_aianaloneorincomb_taes['AES'])}, index=[0])
outputdflist.append(ss)

# Counties AIAN Alone Count Where MDF < HDF
hdfcounties_aianalone = dfhdf[dfhdf['CENRACE'] == 3].group_by('CountyGEOID').agg(HDF_Population = pl.len())
mdfcounties_aianalone = dfmdf[dfmdf['CENRACE'] == 3].group_by('CountyGEOID').agg(MDF_Population = pl.len())
counties_aianalone =  pd.merge(hdfcounties_aianalone, mdfcounties_aianalone, on='GEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
counties_aianalone['MDFltHDF'] = np.where(counties_aianalone['MDF_Population']  < counties_aianalone['HDF_Population'], 1, 0)
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(counties_aianalone),'CountMDFltHDF': np.sum(counties_aianalone['MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(counties_aianalone.loc[counties_aianalone['MDFltHDF'] == 1,'PercDiff'])}, index=[0])
outputdflist.append(ss)
counties_aianalone['AIAN_PopSize'] = pd.cut(counties_aianalone['HDF_Population'], [0,10,100,np.inf], left_closed = True)
for i in counties_aianalone['AIAN_PopSize'].cat.get_categories():
    ss = pd.DataFrame({'Geography':'County', 'Size_Category':'AIAN Population Size {}'.format(i), 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(counties_aianalone[counties_aianalone['AIAN_PopSize'] == i]),'CountMDFltHDF': np.sum(counties_aianalone.loc[counties_aianalone['AIAN_PopSize'] == i, 'MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(counties_aianalone.loc[(counties_aianalone['AIAN_PopSize'] == i)&(counties_aianalone['MDFltHDF'] == 1),'PercDiff'])}, index=[0])
    outputdflist.append(ss)


# Places AIAN Alone Count Where MDF < HDF
hdfplaces_aianalone = dfhdf[dfhdf['CENRACE'] == 3].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
mdfplaces_aianalone = dfmdf[dfmdf['CENRACE'] == 3].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
places_aianalone =  pd.merge(hdfplaces_aianalone, mdfplaces_aianalone, on='IncPlaceGEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
places_aianalone['MDFltHDF'] = np.where(places_aianalone['MDF_Population']  < places_aianalone['HDF_Population'], 1, 0)
ss = pd.DataFrame({'Geography':'Place', 'Size_Category':'All', 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(places_aianalone),'CountMDFltHDF': np.sum(places_aianalone['MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(places_aianalone.loc[places_aianalone['MDFltHDF'] == 1,'PercDiff'])}, index=[0])
outputdflist.append(ss)
places_aianalone['AIAN_PopSize'] = pd.cut(places_aianalone['HDF_Population'], [0,10,100,np.inf], left_closed = True)
for i in places_aianalone['AIAN_PopSize'].cat.get_categories():
    ss = pd.DataFrame({'Geography':'Place', 'Size_Category':'AIAN Population Size {}'.format(i), 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(places_aianalone[places_aianalone['AIAN_PopSize'] == i]),'CountMDFltHDF': np.sum(places_aianalone.loc[places_aianalone['AIAN_PopSize'] == i, 'MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(places_aianalone.loc[(places_aianalone['AIAN_PopSize'] == i)&(places_aianalone['MDFltHDF'] == 1),'PercDiff'])}, index=[0])
    outputdflist.append(ss)


# Fed AIR AIAN Alone Count Where MDF < HDF
hdffedairs_aianalone = dfhdf[dfhdf['CENRACE'] == 3].group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
mdffedairs_aianalone = dfmdf[dfmdf['CENRACE'] == 3].group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
fedairs_aianalone =  pd.merge(hdffedairs_aianalone, mdffedairs_aianalone, on='FedAIRGEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
fedairs_aianalone['MDFltHDF'] = np.where(fedairs_aianalone['MDF_Population']  < fedairs_aianalone['HDF_Population'], 1, 0)
ss = pd.DataFrame({'Geography':'Fed AIR', 'Size_Category':'All', 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(fedairs_aianalone),'CountMDFltHDF': np.sum(fedairs_aianalone['MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(fedairs_aianalone.loc[fedairs_aianalone['MDFltHDF'] == 1,'PercDiff'])}, index=[0])
outputdflist.append(ss)
fedairs_aianalone['AIAN_PopSize'] = pd.cut(fedairs_aianalone['HDF_Population'], [0,10,100,np.inf], left_closed = True)
for i in fedairs_aianalone['AIAN_PopSize'].cat.get_categories():
    ss = pd.DataFrame({'Geography':'Fed AIR', 'Size_Category':'AIAN Population Size {}'.format(i), 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(fedairs_aianalone[fedairs_aianalone['AIAN_PopSize'] == i]),'CountMDFltHDF': np.sum(fedairs_aianalone.loc[fedairs_aianalone['AIAN_PopSize'] == i, 'MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(fedairs_aianalone.loc[(fedairs_aianalone['AIAN_PopSize'] == i)&(fedairs_aianalone['MDFltHDF'] == 1),'PercDiff'])}, index=[0])
    outputdflist.append(ss)


# OTSA AIAN Alone Count Where MDF < HDF
hdfotsas_aianalone = dfhdf[dfhdf['CENRACE'] == 3].group_by(['OTSAGEOID']).agg(HDF_Population = pl.len())
mdfotsas_aianalone = dfmdf[dfmdf['CENRACE'] == 3].group_by(['OTSAGEOID']).agg(MDF_Population = pl.len())
otsas_aianalone =  pd.merge(hdfotsas_aianalone, mdfotsas_aianalone, on='OTSAGEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
otsas_aianalone['MDFltHDF'] = np.where(otsas_aianalone['MDF_Population']  < otsas_aianalone['HDF_Population'], 1, 0)
ss = pd.DataFrame({'Geography':'OTSA', 'Size_Category':'All', 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(otsas_aianalone),'CountMDFltHDF': np.sum(otsas_aianalone['MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(otsas_aianalone.loc[otsas_aianalone['MDFltHDF'] == 1,'PercDiff'])}, index=[0])
outputdflist.append(ss)
otsas_aianalone['AIAN_PopSize'] = pd.cut(otsas_aianalone['HDF_Population'], [0,10,100,np.inf], left_closed = True)
for i in otsas_aianalone['AIAN_PopSize'].cat.get_categories():
    ss = pd.DataFrame({'Geography':'OTSA', 'Size_Category':'AIAN Population Size {}'.format(i), 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(otsas_aianalone[otsas_aianalone['AIAN_PopSize'] == i]),'CountMDFltHDF': np.sum(otsas_aianalone.loc[otsas_aianalone['AIAN_PopSize'] == i, 'MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(otsas_aianalone.loc[(otsas_aianalone['AIAN_PopSize'] == i)&(otsas_aianalone['MDFltHDF'] == 1),'PercDiff'])}, index=[0])
    outputdflist.append(ss)

# ANVSA AIAN Alone Count Where MDF < HDF
hdfanvsas_aianalone = dfhdf[dfhdf['CENRACE'] == 3].group_by(['ANVSAGEOID']).agg(HDF_Population = pl.len())
mdfanvsas_aianalone = dfmdf[dfmdf['CENRACE'] == 3].group_by(['ANVSAGEOID']).agg(MDF_Population = pl.len())
anvsas_aianalone =  pd.merge(hdfanvsas_aianalone, mdfanvsas_aianalone, on='ANVSAGEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
anvsas_aianalone['MDFltHDF'] = np.where(anvsas_aianalone['MDF_Population']  < anvsas_aianalone['HDF_Population'], 1, 0)
ss = pd.DataFrame({'Geography':'ANVSA', 'Size_Category':'All', 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(anvsas_aianalone),'CountMDFltHDF': np.sum(anvsas_aianalone['MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(anvsas_aianalone.loc[anvsas_aianalone['MDFltHDF'] == 1,'PercDiff'])}, index=[0])
outputdflist.append(ss)
anvsas_aianalone['AIAN_PopSize'] = pd.cut(anvsas_aianalone['HDF_Population'], [0,10,100,np.inf], left_closed = True)
for i in anvsas_aianalone['AIAN_PopSize'].cat.get_categories():
    ss = pd.DataFrame({'Geography':'ANVSA', 'Size_Category':'AIAN Population Size {}'.format(i), 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(anvsas_aianalone[anvsas_aianalone['AIAN_PopSize'] == i]),'CountMDFltHDF': np.sum(anvsas_aianalone.loc[anvsas_aianalone['AIAN_PopSize'] == i, 'MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(anvsas_aianalone.loc[(anvsas_aianalone['AIAN_PopSize'] == i)&(anvsas_aianalone['MDFltHDF'] == 1),'PercDiff'])}, index=[0])
    outputdflist.append(ss)

# Counties NHPI Alone Count Where MDF < HDF
hdfcounties_nhpialone = dfhdf[dfhdf['CENRACE'] == 5].group_by('CountyGEOID').agg(HDF_Population = pl.len())
mdfcounties_nhpialone = dfmdf[dfmdf['CENRACE'] == 5].group_by('CountyGEOID').agg(MDF_Population = pl.len())
counties_nhpialone =  pd.merge(hdfcounties_nhpialone, mdfcounties_nhpialone, on='GEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
counties_nhpialone['MDFltHDF'] = np.where(counties_nhpialone['MDF_Population']  < counties_nhpialone['HDF_Population'], 1, 0)
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'NHPI Alone CountMDFltHDF', 'NumCells':len(counties_nhpialone),'CountMDFltHDF': np.sum(counties_nhpialone['MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(counties_nhpialone.loc[counties_nhpialone['MDFltHDF'] == 1,'PercDiff'])}, index=[0])
outputdflist.append(ss)
counties_nhpialone['NHPI_PopSize'] = pd.cut(counties_nhpialone['HDF_Population'], [0,10,100,np.inf], left_closed = True)
for i in counties_nhpialone['NHPI_PopSize'].cat.get_categories():
    ss = pd.DataFrame({'Geography':'County', 'Size_Category':'NHPI Population Size {}'.format(i), 'Characteristic':'NHPI Alone CountMDFltHDF', 'NumCells':len(counties_nhpialone[counties_nhpialone['NHPI_PopSize'] == i]),'CountMDFltHDF': np.sum(counties_nhpialone.loc[counties_nhpialone['NHPI_PopSize'] == i, 'MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(counties_nhpialone.loc[(counties_nhpialone['NHPI_PopSize'] == i)&(counties_nhpialone['MDFltHDF'] == 1),'PercDiff'])}, index=[0])
    outputdflist.append(ss)

# Places NHPI Alone Count Where MDF < HDF
hdfplaces_nhpialone = dfhdf[dfhdf['CENRACE'] == 5].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
mdfplaces_nhpialone = dfmdf[dfmdf['CENRACE'] == 5].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
places_nhpialone =  pd.merge(hdfplaces_nhpialone, mdfplaces_nhpialone, on='IncPlaceGEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
places_nhpialone['MDFltHDF'] = np.where(places_nhpialone['MDF_Population']  < places_nhpialone['HDF_Population'], 1, 0)
ss = pd.DataFrame({'Geography':'Place', 'Size_Category':'All', 'Characteristic':'NHPI Alone CountMDFltHDF', 'NumCells':len(places_nhpialone),'CountMDFltHDF': np.sum(places_nhpialone['MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(places_nhpialone.loc[places_nhpialone['MDFltHDF'] == 1,'PercDiff'])}, index=[0])
outputdflist.append(ss)
places_nhpialone['NHPI_PopSize'] = pd.cut(places_nhpialone['HDF_Population'], [0,10,100,np.inf], left_closed = True)
for i in places_nhpialone['NHPI_PopSize'].cat.get_categories():
    ss = pd.DataFrame({'Geography':'Place', 'Size_Category':'NHPI Population Size {}'.format(i), 'Characteristic':'NHPI Alone CountMDFltHDF', 'NumCells':len(places_nhpialone[places_nhpialone['NHPI_PopSize'] == i]),'CountMDFltHDF': np.sum(places_nhpialone.loc[places_nhpialone['NHPI_PopSize'] == i, 'MDFltHDF']), 'MedianPctDiffWhereMDFltHDF':np.nanmedian(places_nhpialone.loc[(places_nhpialone['NHPI_PopSize'] == i)&(places_nhpialone['MDFltHDF'] == 1),'PercDiff'])}, index=[0])
    outputdflist.append(ss)

# Tracts AIAN Alone or in Combination 
hdftracts_aianincomb = dfhdf[(dfhdf['aianalone-or-incomb']==1)].group_by(['TractGEOID']).agg(HDF_Population = pl.len())
mdftracts_aianincomb = dfmdf[(dfmdf['aianalone-or-incomb']==1)].group_by(['TractGEOID']).agg(MDF_Population = pl.len())
tracts_aianincomb =  pd.merge(hdftracts_aianincomb, mdftracts_aianincomb, on='GEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
tracts_aianincomb['HundredPlusMDFLessThan20HDF'] = np.where((tracts_aianincomb['HDF_Population'] < 20) & (tracts_aianincomb['MDF_Population'] >=100), 1, 0)
tracts_aianincomb['LessThan20MDFHundredPlusHDF'] = np.where((tracts_aianincomb['HDF_Population'] >=100) & (tracts_aianincomb['MDF_Population'] < 20), 1, 0)
ss = tracts_aianincomb.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "AIAN Alone Or In Combination")
ss = ss.join(pd.DataFrame({'Number100PlusMDFLessThan20HDF':np.sum(tracts_aianincomb['HundredPlusMDFLessThan20HDF']), 'NumberLessThan20MDF100PlusHDF':np.sum(tracts_aianincomb['LessThan20MDFHundredPlusHDF'])}, index=[0]))
outputdflist.append(ss)

# Tracts NHPI
hdftracts_nhpialone = dfhdf[dfhdf['CENRACE'] == 5].group_by(['TractGEOID']).agg(HDF_Population = pl.len())
mdftracts_nhpialone = dfmdf[dfmdf['CENRACE'] == 5].group_by(['TractGEOID']).agg(MDF_Population = pl.len())
tracts_nhpialone =  pd.merge(hdftracts_nhpialone, mdftracts_nhpialone, on='GEOID', how = 'outer', validate = mergeValidation).pipe(calculate_stats)
tracts_nhpialone['HundredPlusMDFLessThan20HDF'] = np.where((tracts_nhpialone['HDF_Population'] < 20) & (tracts_nhpialone['MDF_Population'] >=100), 1, 0)
tracts_nhpialone['LessThan20MDFHundredPlusHDF'] = np.where((tracts_nhpialone['HDF_Population'] >=100) & (tracts_nhpialone['MDF_Population'] < 20), 1, 0)
ss = tracts_nhpialone.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "NHPI Alone")
ss = ss.join(pd.DataFrame({'Number100PlusMDFLessThan20HDF':np.sum(tracts_nhpialone['HundredPlusMDFLessThan20HDF']), 'NumberLessThan20MDF100PlusHDF':np.sum(tracts_nhpialone['LessThan20MDFHundredPlusHDF'])}, index=[0]))
outputdflist.append(ss)

# Counties Total Population Cross 50000
hdfcounties_totalpop = dfhdf().group_by('CountyGEOID').agg(HDF_Population = pl.len())
mdfcounties_totalpop = dfmdf().group_by('CountyGEOID').agg(MDF_Population = pl.len())
counties_totalpop =  pd.merge(hdfcounties_totalpop, mdfcounties_totalpop, on='GEOID', how = 'outer', validate = mergeValidation)
counties_totalpop['HDFgt50kMDFlt50k'] = np.where((counties_totalpop['MDF_Population'] < 50000) & (counties_totalpop['HDF_Population'] > 50000), 1, 0)
counties_totalpop['HDFlt50kMDFgt50k'] = np.where((counties_totalpop['MDF_Population'] > 50000) & (counties_totalpop['HDF_Population'] < 50000), 1, 0)
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'Cross 50000','NumCells': len(counties_totalpop),'NumberHDFgt50kMDFlt50k': np.sum(counties_totalpop['HDFgt50kMDFlt50k']), 'NumberHDFlt50kMDFgt50k': np.sum(counties_totalpop['HDFlt50kMDFgt50k'])}, index=[0])
outputdflist.append(ss)

# Places Total Population Cross 50000
hdfplaces_totalpop = dfhdf().group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
mdfplaces_totalpop = dfmdf().group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
places_totalpop =  pd.merge(hdfplaces_totalpop, mdfplaces_totalpop, on='IncPlaceGEOID', how = 'outer', validate = mergeValidation)
places_totalpop['HDFgt50kMDFlt50k'] = np.where((places_totalpop['MDF_Population'] < 50000) & (places_totalpop['HDF_Population'] > 50000), 1, 0)
places_totalpop['HDFlt50kMDFgt50k'] = np.where((places_totalpop['MDF_Population'] > 50000) & (places_totalpop['HDF_Population'] < 50000), 1, 0)
ss = pd.DataFrame({'Geography':'Place', 'Size_Category':'All', 'Characteristic':'Cross 50000','NumCells': len(places_totalpop),'NumberHDFgt50kMDFlt50k': np.sum(places_totalpop['HDFgt50kMDFlt50k']), 'NumberHDFlt50kMDFgt50k': np.sum(places_totalpop['HDFlt50kMDFgt50k'])}, index=[0])
outputdflist.append(ss)

# Tracts Total Population Cross 50000
hdftracts_totalpop = dfhdf().group_by('TractGEOID').agg(HDF_Population = pl.len())
mdftracts_totalpop = dfmdf().group_by('TractGEOID').agg(MDF_Population = pl.len())
tracts_totalpop =  pd.merge(hdftracts_totalpop, mdftracts_totalpop, on='GEOID', how = 'outer', validate = mergeValidation)
tracts_totalpop['HDFgt50kMDFlt50k'] = np.where((tracts_totalpop['MDF_Population'] < 50000) & (tracts_totalpop['HDF_Population'] > 50000), 1, 0)
tracts_totalpop['HDFlt50kMDFgt50k'] = np.where((tracts_totalpop['MDF_Population'] > 50000) & (tracts_totalpop['HDF_Population'] < 50000), 1, 0)
ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'Cross 50000','NumCells': len(tracts_totalpop),'NumberHDFgt50kMDFlt50k': np.sum(tracts_totalpop['HDFgt50kMDFlt50k']), 'NumberHDFlt50kMDFgt50k': np.sum(tracts_totalpop['HDFlt50kMDFgt50k'])}, index=[0])
outputdflist.append(ss)


print("{} Use Cases Done".format(datetime.now()))

print("{} Starting Improbable and Impossible Measurements".format(datetime.now()))

# States Total Population Should Be Equal
hdfstates_totalpop = dfhdf().group_by(['TABBLKST']).agg(HDF_Population = pl.len())
mdfstates_totalpop = dfmdf().group_by(['TABBLKST']).agg(MDF_Population = pl.len())
states_totalpop =  pd.merge(hdfstates_totalpop, mdfstates_totalpop, on='GEOID', how = 'outer', validate = mergeValidation)
states_totalpop['HDFneMDF'] = np.where(states_totalpop['MDF_Population'] != states_totalpop['HDF_Population'], 1, 0)
ss = pd.DataFrame({'Geography':'State', 'Size_Category':'All', 'Characteristic':'Total Population','NumCells': len(states_totalpop),'NumberHDFneMDF': np.sum(states_totalpop['HDFneMDF'])}, index=[0])
outputdflist.append(ss)

# Counties with at least 5 children under age 5 and no women age 18 through 44
hdfcounties_poplt5 = dfhdf[dfhdf['QAGE'] < 5].group_by('CountyGEOID').size().reset_index(name='HDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
mdfcounties_poplt5 = dfmdf[dfmdf['QAGE'] < 5].group_by('CountyGEOID').size().reset_index(name='MDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
hdfcounties_popfem1844 = dfhdf[(dfhdf['QSEX'] == '2')&(dfhdf['QAGE'] >= 18)&(dfhdf['QAGE'] < 45)].group_by('CountyGEOID').size().reset_index(name='HDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
mdfcounties_popfem1844 = dfmdf[(dfmdf['QSEX'] == '2')&(dfmdf['QAGE'] >= 18)&(dfmdf['QAGE'] < 45)].group_by('CountyGEOID').size().reset_index(name='MDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])

hdfcounties_poplt5 = hdfcounties_poplt5[hdfcounties_poplt5['HDF_Children'] >=5]
mdfcounties_poplt5 = mdfcounties_poplt5[mdfcounties_poplt5['MDF_Children'] >=5]

hdfcounties =  pd.merge(hdfcounties_poplt5, hdfcounties_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)
mdfcounties =  pd.merge(mdfcounties_poplt5, mdfcounties_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)

hdfcounties['ChildrenNoMoms'] = np.where((hdfcounties['HDF_Children'] >= 5)&(hdfcounties['HDF_MomAge'] == 0), 1, 0)
mdfcounties['ChildrenNoMoms'] = np.where((mdfcounties['MDF_Children'] >= 5)&(mdfcounties['MDF_MomAge'] == 0), 1, 0)

ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'HDF Children No Moms','NumCells':len(hdfcounties), 'Inconsistent':np.sum(hdfcounties['ChildrenNoMoms'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'MDF Children No Moms','NumCells':len(mdfcounties), 'Inconsistent':np.sum(mdfcounties['ChildrenNoMoms'])}, index=[0])
outputdflist.append(ss)

# Tracts with at least 5 children under age 5 and no women age 18 through 44
hdftracts_poplt5 = dfhdf[dfhdf['QAGE'] < 5].group_by('TractGEOID').size().reset_index(name='HDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
mdftracts_poplt5 = dfmdf[dfmdf['QAGE'] < 5].group_by('TractGEOID').size().reset_index(name='MDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
hdftracts_popfem1844 = dfhdf[(dfhdf['QSEX'] == '2')&(dfhdf['QAGE'] >= 18)&(dfhdf['QAGE'] < 45)].group_by('TractGEOID').size().reset_index(name='HDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
mdftracts_popfem1844 = dfmdf[(dfmdf['QSEX'] == '2')&(dfmdf['QAGE'] >= 18)&(dfmdf['QAGE'] < 45)].group_by('TractGEOID').size().reset_index(name='MDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()

hdftracts_poplt5 = hdftracts_poplt5[hdftracts_poplt5['HDF_Children'] >=5]
mdftracts_poplt5 = mdftracts_poplt5[mdftracts_poplt5['MDF_Children'] >=5]

hdftracts =  pd.merge(hdftracts_poplt5, hdftracts_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)
mdftracts =  pd.merge(mdftracts_poplt5, mdftracts_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)

hdftracts['ChildrenNoMoms'] = np.where((hdftracts['HDF_Children'] >= 5)&(hdftracts['HDF_MomAge'] == 0), 1, 0)
mdftracts['ChildrenNoMoms'] = np.where((mdftracts['MDF_Children'] >= 5)&(mdftracts['MDF_MomAge'] == 0), 1, 0)

ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF Children No Moms','NumCells':len(hdftracts), 'Inconsistent':np.sum(hdftracts['ChildrenNoMoms'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF Children No Moms','NumCells':len(mdftracts), 'Inconsistent':np.sum(mdftracts['ChildrenNoMoms'])}, index=[0])
outputdflist.append(ss)

# Counties with at least 5 children under age 5 and no women age 18 through 44 by race alone
for r in racealonecats:
    hdfcounties_poplt5 = dfhdf[(dfhdf['RACEALONE'] == r)&(dfhdf['QAGE'] < 5)].group_by('CountyGEOID').size().reset_index(name='HDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
    mdfcounties_poplt5 = dfmdf[(dfmdf['RACEALONE'] == r)&(dfmdf['QAGE'] < 5)].group_by('CountyGEOID').size().reset_index(name='MDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
    hdfcounties_popfem1844 = dfhdf[(dfhdf['RACEALONE'] == r)&(dfhdf['QSEX'] == '2')&(dfhdf['QAGE'] >= 18)&(dfhdf['QAGE'] < 45)].group_by('CountyGEOID').size().reset_index(name='HDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
    mdfcounties_popfem1844 = dfmdf[(dfmdf['RACEALONE'] == r)&(dfmdf['QSEX'] == '2')&(dfmdf['QAGE'] >= 18)&(dfmdf['QAGE'] < 45)].group_by('CountyGEOID').size().reset_index(name='MDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])

    hdfcounties_poplt5 = hdfcounties_poplt5[hdfcounties_poplt5['HDF_Children'] >=5]
    mdfcounties_poplt5 = mdfcounties_poplt5[mdfcounties_poplt5['MDF_Children'] >=5]

    hdfcounties =  pd.merge(hdfcounties_poplt5, hdfcounties_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)
    mdfcounties =  pd.merge(mdfcounties_poplt5, mdfcounties_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)

    hdfcounties['ChildrenNoMoms'] = np.where((hdfcounties['HDF_Children'] >= 5)&(hdfcounties['HDF_MomAge'] == 0), 1, 0)
    mdfcounties['ChildrenNoMoms'] = np.where((mdfcounties['MDF_Children'] >= 5)&(mdfcounties['MDF_MomAge'] == 0), 1, 0)

    ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"HDF Children No Moms {race}".format(race = racealonedict.get(r)),'NumCells':len(hdfcounties), 'Inconsistent':np.sum(hdfcounties['ChildrenNoMoms'])}, index=[0])
    outputdflist.append(ss)
    ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"MDF Children No Moms {race}".format(race = racealonedict.get(r)),'NumCells':len(mdfcounties), 'Inconsistent':np.sum(mdfcounties['ChildrenNoMoms'])}, index=[0])
    outputdflist.append(ss)

# Counties with at least 5 children under age 5 and no women age 18 through 44 Hispanic
hdfcounties_poplt5 = dfhdf[(dfhdf['CENHISP'] == '2')&(dfhdf['QAGE'] < 5)].group_by('CountyGEOID').size().reset_index(name='HDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
mdfcounties_poplt5 = dfmdf[(dfmdf['CENHISP'] == '2')&(dfmdf['QAGE'] < 5)].group_by('CountyGEOID').size().reset_index(name='MDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
hdfcounties_popfem1844 = dfhdf[(dfhdf['CENHISP'] == '2')&(dfhdf['QSEX'] == '2')&(dfhdf['QAGE'] >= 18)&(dfhdf['QAGE'] < 45)].group_by('CountyGEOID').size().reset_index(name='HDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
mdfcounties_popfem1844 = dfmdf[(dfmdf['CENHISP'] == '2')&(dfmdf['QSEX'] == '2')&(dfmdf['QAGE'] >= 18)&(dfmdf['QAGE'] < 45)].group_by('CountyGEOID').size().reset_index(name='MDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])

hdfcounties_poplt5 = hdfcounties_poplt5[hdfcounties_poplt5['HDF_Children'] >=5]
mdfcounties_poplt5 = mdfcounties_poplt5[mdfcounties_poplt5['MDF_Children'] >=5]

hdfcounties =  pd.merge(hdfcounties_poplt5, hdfcounties_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)
mdfcounties =  pd.merge(mdfcounties_poplt5, mdfcounties_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)

hdfcounties['ChildrenNoMoms'] = np.where((hdfcounties['HDF_Children'] >= 5)&(hdfcounties['HDF_MomAge'] == 0), 1, 0)
mdfcounties['ChildrenNoMoms'] = np.where((mdfcounties['MDF_Children'] >= 5)&(mdfcounties['MDF_MomAge'] == 0), 1, 0)

ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'HDF Children No Moms Hispanic','NumCells':len(hdfcounties), 'Inconsistent':np.sum(hdfcounties['ChildrenNoMoms'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'MDF Children No Moms Hispanic','NumCells':len(mdfcounties), 'Inconsistent':np.sum(mdfcounties['ChildrenNoMoms'])}, index=[0])
outputdflist.append(ss)

# Counties with at least 5 children under age 5 and no women age 18 through 44 Non-Hispanic White
hdfcounties_poplt5 = dfhdf[(dfhdf['CENHISP'] == '1')&(dfhdf['RACEALONE'] == 1)&(dfhdf['QAGE'] < 5)].group_by('CountyGEOID').size().reset_index(name='HDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
mdfcounties_poplt5 = dfmdf[(dfmdf['CENHISP'] == '1')&(dfmdf['RACEALONE'] == 1)&(dfmdf['QAGE'] < 5)].group_by('CountyGEOID').size().reset_index(name='MDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
hdfcounties_popfem1844 = dfhdf[(dfhdf['CENHISP'] == '1')&(dfhdf['RACEALONE'] == 1)&(dfhdf['QSEX'] == '2')&(dfhdf['QAGE'] >= 18)&(dfhdf['QAGE'] < 45)].group_by('CountyGEOID').size().reset_index(name='HDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
mdfcounties_popfem1844 = dfmdf[(dfmdf['CENHISP'] == '1')&(dfmdf['RACEALONE'] == 1)&(dfmdf['QSEX'] == '2')&(dfmdf['QAGE'] >= 18)&(dfmdf['QAGE'] < 45)].group_by('CountyGEOID').size().reset_index(name='MDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])

hdfcounties_poplt5 = hdfcounties_poplt5[hdfcounties_poplt5['HDF_Children'] >=5]
mdfcounties_poplt5 = mdfcounties_poplt5[mdfcounties_poplt5['MDF_Children'] >=5]

hdfcounties =  pd.merge(hdfcounties_poplt5, hdfcounties_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)
mdfcounties =  pd.merge(mdfcounties_poplt5, mdfcounties_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)

hdfcounties['ChildrenNoMoms'] = np.where((hdfcounties['HDF_Children'] >= 5)&(hdfcounties['HDF_MomAge'] == 0), 1, 0)
mdfcounties['ChildrenNoMoms'] = np.where((mdfcounties['MDF_Children'] >= 5)&(mdfcounties['MDF_MomAge'] == 0), 1, 0)

ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'HDF Children No Moms Non-Hispanic White','NumCells':len(hdfcounties), 'Inconsistent':np.sum(hdfcounties['ChildrenNoMoms'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'MDF Children No Moms Non-Hispanic White','NumCells':len(mdfcounties), 'Inconsistent':np.sum(mdfcounties['ChildrenNoMoms'])}, index=[0])
outputdflist.append(ss)

# Tracts with at least 5 children under age 5 and no women age 18 through 44 by race alone
for r in racealonecats:
    hdftracts_poplt5 = dfhdf[(dfhdf['RACEALONE'] == r)&(dfhdf['QAGE'] < 5)].group_by('TractGEOID').size().reset_index(name='HDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
    mdftracts_poplt5 = dfmdf[(dfmdf['RACEALONE'] == r)&(dfmdf['QAGE'] < 5)].group_by('TractGEOID').size().reset_index(name='MDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
    hdftracts_popfem1844 = dfhdf[(dfhdf['RACEALONE'] == r)&(dfhdf['QSEX'] == '2')&(dfhdf['QAGE'] >= 18)&(dfhdf['QAGE'] < 45)].group_by('TractGEOID').size().reset_index(name='HDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
    mdftracts_popfem1844 = dfmdf[(dfmdf['RACEALONE'] == r)&(dfmdf['QSEX'] == '2')&(dfmdf['QAGE'] >= 18)&(dfmdf['QAGE'] < 45)].group_by('TractGEOID').size().reset_index(name='MDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
    
    hdftracts_poplt5 = hdftracts_poplt5[hdftracts_poplt5['HDF_Children'] >=5]
    mdftracts_poplt5 = mdftracts_poplt5[mdftracts_poplt5['MDF_Children'] >=5]

    hdftracts =  pd.merge(hdftracts_poplt5, hdftracts_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)
    mdftracts =  pd.merge(mdftracts_poplt5, mdftracts_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)

    hdftracts['ChildrenNoMoms'] = np.where((hdftracts['HDF_Children'] >= 5)&(hdftracts['HDF_MomAge'] == 0), 1, 0)
    mdftracts['ChildrenNoMoms'] = np.where((mdftracts['MDF_Children'] >= 5)&(mdftracts['MDF_MomAge'] == 0), 1, 0)

    ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':"HDF Children No Moms {race}".format(race = racealonedict.get(r)),'NumCells':len(hdftracts), 'Inconsistent':np.sum(hdftracts['ChildrenNoMoms'])}, index=[0])
    outputdflist.append(ss)
    ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':"MDF Children No Moms {race}".format(race = racealonedict.get(r)),'NumCells':len(mdftracts), 'Inconsistent':np.sum(mdftracts['ChildrenNoMoms'])}, index=[0])
    outputdflist.append(ss)

# Tracts with at least 5 children under age 5 and no women age 18 through 44 Hispanic
hdftracts_poplt5 = dfhdf[(dfhdf['CENHISP'] == '2')&(dfhdf['QAGE'] < 5)].group_by('TractGEOID').size().reset_index(name='HDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
mdftracts_poplt5 = dfmdf[(dfmdf['CENHISP'] == '2')&(dfmdf['QAGE'] < 5)].group_by('TractGEOID').size().reset_index(name='MDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
hdftracts_popfem1844 = dfhdf[(dfhdf['CENHISP'] == '2')&(dfhdf['QSEX'] == '2')&(dfhdf['QAGE'] >= 18)&(dfhdf['QAGE'] < 45)].group_by('TractGEOID').size().reset_index(name='HDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
mdftracts_popfem1844 = dfmdf[(dfmdf['CENHISP'] == '2')&(dfmdf['QSEX'] == '2')&(dfmdf['QAGE'] >= 18)&(dfmdf['QAGE'] < 45)].group_by('TractGEOID').size().reset_index(name='MDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()

hdftracts_poplt5 = hdftracts_poplt5[hdftracts_poplt5['HDF_Children'] >=5]
mdftracts_poplt5 = mdftracts_poplt5[mdftracts_poplt5['MDF_Children'] >=5]

hdftracts =  pd.merge(hdftracts_poplt5, hdftracts_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)
mdftracts =  pd.merge(mdftracts_poplt5, mdftracts_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)

hdftracts['ChildrenNoMoms'] = np.where((hdftracts['HDF_Children'] >= 5)&(hdftracts['HDF_MomAge'] == 0), 1, 0)
mdftracts['ChildrenNoMoms'] = np.where((mdftracts['MDF_Children'] >= 5)&(mdftracts['MDF_MomAge'] == 0), 1, 0)

ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF Children No Moms Hispanic','NumCells':len(hdftracts), 'Inconsistent':np.sum(hdftracts['ChildrenNoMoms'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF Children No Moms Hispanic','NumCells':len(mdftracts), 'Inconsistent':np.sum(mdftracts['ChildrenNoMoms'])}, index=[0])
outputdflist.append(ss)

# Tracts with at least 5 children under age 5 and no women age 18 through 44 Non-Hispanic White
hdftracts_poplt5 = dfhdf[(dfhdf['RACEALONE'] == 1)&(dfhdf['CENHISP'] == '1')&(dfhdf['QAGE'] < 5)].group_by('TractGEOID').size().reset_index(name='HDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
mdftracts_poplt5 = dfmdf[(dfmdf['RACEALONE'] == 1)&(dfmdf['CENHISP'] == '1')&(dfmdf['QAGE'] < 5)].group_by('TractGEOID').size().reset_index(name='MDF_Children').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
hdftracts_popfem1844 = dfhdf[(dfhdf['RACEALONE'] == 1)&(dfhdf['CENHISP'] == '1')&(dfhdf['QSEX'] == '2')&(dfhdf['QAGE'] >= 18)&(dfhdf['QAGE'] < 45)].group_by('TractGEOID').size().reset_index(name='HDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
mdftracts_popfem1844 = dfmdf[(dfmdf['RACEALONE'] == 1)&(dfmdf['CENHISP'] == '1')&(dfmdf['QSEX'] == '2')&(dfmdf['QAGE'] >= 18)&(dfmdf['QAGE'] < 45)].group_by('TractGEOID').size().reset_index(name='MDF_MomAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()

hdftracts_poplt5 = hdftracts_poplt5[hdftracts_poplt5['HDF_Children'] >=5]
mdftracts_poplt5 = mdftracts_poplt5[mdftracts_poplt5['MDF_Children'] >=5]

hdftracts =  pd.merge(hdftracts_poplt5, hdftracts_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)
mdftracts =  pd.merge(mdftracts_poplt5, mdftracts_popfem1844, on='GEOID', how = 'left', validate = mergeValidation)

hdftracts['ChildrenNoMoms'] = np.where((hdftracts['HDF_Children'] >= 5)&(hdftracts['HDF_MomAge'] == 0), 1, 0)
mdftracts['ChildrenNoMoms'] = np.where((mdftracts['MDF_Children'] >= 5)&(mdftracts['MDF_MomAge'] == 0), 1, 0)

ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF Children No Moms Non-Hispanic White','NumCells':len(hdftracts), 'Inconsistent':np.sum(hdftracts['ChildrenNoMoms'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF Children No Moms Non-Hispanic White','NumCells':len(mdftracts), 'Inconsistent':np.sum(mdftracts['ChildrenNoMoms'])}, index=[0])
outputdflist.append(ss)

# Tracts with at least 5 people and all of the same sex
hdftracts_males = dfhdf[(dfhdf['QSEX'] == '1')].group_by('TractGEOID').size().reset_index(name='HDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
mdftracts_males = dfmdf[(dfmdf['QSEX'] == '1')].group_by('TractGEOID').size().reset_index(name='MDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
hdftracts_females = dfhdf[(dfhdf['QSEX'] == '2')].group_by('TractGEOID').size().reset_index(name='HDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
mdftracts_females = dfmdf[(dfmdf['QSEX'] == '2')].group_by('TractGEOID').size().reset_index(name='MDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()

hdftracts =  pd.merge(hdftracts_males, hdftracts_females, on='GEOID', how = 'outer', validate = mergeValidation)
mdftracts =  pd.merge(mdftracts_males, mdftracts_females, on='GEOID', how = 'outer', validate = mergeValidation)

hdftracts['Total'] = hdftracts['HDF_Females'] + hdftracts['HDF_Males']
mdftracts['Total'] = mdftracts['MDF_Females'] + mdftracts['MDF_Males']

hdftracts = hdftracts[hdftracts['Total'] >=5]
mdftracts = mdftracts[mdftracts['Total'] >=5]

hdftracts['AllSameSex'] = np.where((hdftracts['HDF_Males'] == 0)|(hdftracts['HDF_Females'] == 0), 1, 0)
mdftracts['AllSameSex'] = np.where((mdftracts['MDF_Males'] == 0)|(mdftracts['MDF_Females'] == 0), 1, 0)

ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF All Same Sex','NumCells':len(hdftracts), 'Inconsistent':np.sum(hdftracts['AllSameSex'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF All Same Sex','NumCells':len(mdftracts), 'Inconsistent':np.sum(mdftracts['AllSameSex'])}, index=[0])
outputdflist.append(ss)

# Tracts with at least one of the single years of age between 0 and 17 by sex has a zero count
hdftracts_totunder18 = dfhdf[(dfhdf['QAGE'] < 18)].group_by('TractGEOID').size().reset_index(name='HDF_TotUnder18').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()
mdftracts_totunder18 = dfmdf[(dfmdf['QAGE'] < 18)].group_by('TractGEOID').size().reset_index(name='MDF_TotUnder18').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(alltractsindex, fill_value=0).reset_index()

hdftracts_totunder18gt200 = hdftracts_totunder18.loc[hdftracts_totunder18['HDF_TotUnder18'] > 200,'GEOID'].tolist()
mdftracts_totunder18gt200 = mdftracts_totunder18.loc[mdftracts_totunder18['MDF_TotUnder18'] > 200,'GEOID'].tolist()

hdftract_1yageunder18_index = pd.MultiIndex.from_product([hdftracts_totunder18gt200, list(range(0,18))], names = ['GEOID','QAGE'])
mdftract_1yageunder18_index = pd.MultiIndex.from_product([mdftracts_totunder18gt200, list(range(0,18))], names = ['GEOID','QAGE'])

hdftracts_under18 = dfhdf[(dfhdf['QAGE'] < 18)].group_by(['TABBLKST','TABBLKCOU','TABTRACTCE','QAGE']).agg(HDF_Population = pl.len())
mdftracts_under18 = dfmdf[(dfmdf['QAGE'] < 18)].group_by(['TABBLKST','TABBLKCOU','TABTRACTCE','QAGE']).agg(MDF_Population = pl.len())

hdftracts_under18['ZeroAge'] = np.where((hdftracts_under18['HDF_Population'] == 0), 1, 0)
mdftracts_under18['ZeroAge'] = np.where((mdftracts_under18['MDF_Population'] == 0), 1, 0)

hdftracts_anyzeros = hdftracts_under18.group_by('GEOID')['ZeroAge'].max()
mdftracts_anyzeros = mdftracts_under18.group_by('GEOID')['ZeroAge'].max()

ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF Zero Age','NumCells':len(hdftracts_anyzeros), 'Inconsistent':np.sum(hdftracts_anyzeros)}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF Zero Age','NumCells':len(mdftracts_anyzeros), 'Inconsistent':np.sum(mdftracts_anyzeros)}, index=[0])
outputdflist.append(ss)

hdftracts_under18.to_csv(f"{OUTPUTDIR}/hdftracts_under18.csv", index=False)
hdftracts_anyzeros.to_csv(f"{OUTPUTDIR}/hdftracts_under18anyzeros.csv", index=False)

# Blocks with population all 17 or younger 
hdfblocks_gqpop = dfhdf[dfhdf['RTYPE']=="5"].group_by(['TABBLKST','TABBLKCOU','TABTRACTCE', 'TABBLK']).size().reset_index(name='HDF_GQPopulation').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE + x.TABBLK).drop(columns = ['TABBLKST', 'TABBLKCOU', 'TABTRACTCE', 'TABBLK']).set_index('GEOID').reindex(allblocksindex, fill_value=0).reset_index()
mdfblocks_gqpop = dfmdf[dfmdf['RTYPE']=="5"].group_by(['TABBLKST','TABBLKCOU','TABTRACTCE', 'TABBLK']).size().reset_index(name='MDF_GQPopulation').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE + x.TABBLK).drop(columns = ['TABBLKST', 'TABBLKCOU', 'TABTRACTCE', 'TABBLK']).set_index('GEOID').reindex(allblocksindex, fill_value=0).reset_index()

hdfblocks_nogqs = hdfblocks_gqpop.loc[hdfblocks_gqpop['HDF_GQPopulation'] == 0,'GEOID'].tolist()
mdfblocks_nogqs = mdfblocks_gqpop.loc[mdfblocks_gqpop['MDF_GQPopulation'] == 0,'GEOID'].tolist()

del mdfblocks_gqpop
del hdfblocks_gqpop

hdfblocks_allpop = dfhdf().group_by(['TABBLKST','TABBLKCOU','TABTRACTCE', 'TABBLK']).agg(HDF_Population = pl.len())
mdfblocks_allpop = dfmdf().group_by(['TABBLKST','TABBLKCOU','TABTRACTCE', 'TABBLK']).agg(MDF_Population = pl.len())

hdfblocks_somepop = hdfblocks_allpop.loc[hdfblocks_allpop['HDF_Population'] > 0,'GEOID'].tolist()
mdfblocks_somepop = mdfblocks_allpop.loc[mdfblocks_allpop['MDF_Population'] > 0,'GEOID'].tolist()

del hdfblocks_allpop
del mdfblocks_allpop

hdfblocks_nogqs_somepop = set(hdfblocks_nogqs).intersection(hdfblocks_somepop)
mdfblocks_nogqs_somepop = set(mdfblocks_nogqs).intersection(mdfblocks_somepop)

hdfblocks_nogqs_index = pd.Index(hdfblocks_nogqs_somepop, name='BlockGEOID')
mdfblocks_nogqs_index = pd.Index(mdfblocks_nogqs_somepop, name='BlockGEOID')

del hdfblocks_nogqs
del mdfblocks_nogqs
del hdfblocks_somepop
del mdfblocks_somepop

print("HDF Blocks with >0 Population and No GQ Population", len(hdfblocks_nogqs_somepop))
print("MDF Blocks with >0 Population and No GQ Population", len(mdfblocks_nogqs_somepop))

del hdfblocks_nogqs_somepop
del mdfblocks_nogqs_somepop

hdfblocks_totpop = dfhdf().group_by(['BlockGEOID']).agg(HDF_Population = pl.len())
mdfblocks_totpop = dfmdf().group_by(['BlockGEOID']).agg(MDF_Population = pl.len())

hdfblocks_18 = dfhdf[dfhdf['QAGE'] < 18].group_by(['BlockGEOID']).size().reset_index(name='HDF_Under18').set_index('BlockGEOID').reindex(hdfblocks_nogqs_index, fill_value=0).reset_index()
mdfblocks_18 = dfmdf[dfmdf['QAGE'] < 18].group_by(['BlockGEOID']).size().reset_index(name='MDF_Under18').set_index('BlockGEOID').reindex(mdfblocks_nogqs_index, fill_value=0).reset_index()

hdfblocks =  pd.merge(hdfblocks_totpop, hdfblocks_18, on='BlockGEOID', how = 'inner', validate = mergeValidation)
mdfblocks =  pd.merge(mdfblocks_totpop, mdfblocks_18, on='BlockGEOID', how = 'inner', validate = mergeValidation)

del hdfblocks_18
del mdfblocks_18
del hdfblocks_totpop
del mdfblocks_totpop

hdfblocks['Zero18andOver'] = np.where((hdfblocks['HDF_Population'] > 0)&(hdfblocks['HDF_Under18'] == hdfblocks['HDF_Population']), 1, 0)
mdfblocks['Zero18andOver'] = np.where((mdfblocks['MDF_Population'] > 0)&(mdfblocks['MDF_Under18'] == mdfblocks['MDF_Population']), 1, 0)

ss = pd.DataFrame({'Geography':'Block', 'Size_Category':'All', 'Characteristic':'HDF Everyone Under 18','NumCells':len(hdfblocks), 'Inconsistent':np.sum(hdfblocks['Zero18andOver'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'Block', 'Size_Category':'All', 'Characteristic':'MDF Everyone Under 18','NumCells':len(mdfblocks), 'Inconsistent':np.sum(mdfblocks['Zero18andOver'])}, index=[0])
outputdflist.append(ss)

del hdfblocks
del mdfblocks
del hdfblocks_nogqs_index
del mdfblocks_nogqs_index


# Counties where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women
hdfcounties_males = dfhdf[(dfhdf['QSEX'] == '1')].group_by('CountyGEOID').size().reset_index(name='HDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
mdfcounties_males = dfmdf[(dfmdf['QSEX'] == '1')].group_by('CountyGEOID').size().reset_index(name='MDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
hdfcounties_females = dfhdf[(dfhdf['QSEX'] == '2')].group_by('CountyGEOID').size().reset_index(name='HDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
mdfcounties_females = dfmdf[(dfmdf['QSEX'] == '2')].group_by('CountyGEOID').size().reset_index(name='MDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])

hdfcounties_gt5males = hdfcounties_males.loc[hdfcounties_males['HDF_Males'] >=5, 'GEOID'].tolist()
mdfcounties_gt5males = mdfcounties_males.loc[mdfcounties_males['MDF_Males'] >=5, 'GEOID'].tolist()
hdfcounties_gt5females = hdfcounties_females.loc[hdfcounties_females['HDF_Females'] >=5, 'GEOID'].tolist()
mdfcounties_gt5females = mdfcounties_females.loc[mdfcounties_females['MDF_Females'] >=5, 'GEOID'].tolist()

hdfcounties_gt5bothsex =  list(set(hdfcounties_gt5males).intersection(hdfcounties_gt5females))
mdfcounties_gt5bothsex =  list(set(mdfcounties_gt5males).intersection(mdfcounties_gt5females))

hdfcounties_gt5bothsex_index = pd.Index(hdfcounties_gt5bothsex, name='GEOID')
mdfcounties_gt5bothsex_index = pd.Index(mdfcounties_gt5bothsex, name='GEOID')

hdfcounties_malemedage = dfhdf[(dfhdf['QSEX'] == '1')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(hdfcounties_gt5bothsex_index).reset_index()
mdfcounties_malemedage = dfmdf[(dfmdf['QSEX'] == '1')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(mdfcounties_gt5bothsex_index).reset_index()
hdfcounties_femalemedage = dfhdf[(dfhdf['QSEX'] == '2')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(hdfcounties_gt5bothsex_index).reset_index()
mdfcounties_femalemedage = dfmdf[(dfmdf['QSEX'] == '2')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(mdfcounties_gt5bothsex_index).reset_index()

hdfcounties =  pd.merge(hdfcounties_malemedage, hdfcounties_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)
mdfcounties =  pd.merge(mdfcounties_malemedage, mdfcounties_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)

hdfcounties['BigAgeDiff'] = np.where(np.abs(hdfcounties['HDF_MaleAge'] - hdfcounties['HDF_FemaleAge']) >= 20, 1, 0)
mdfcounties['BigAgeDiff'] = np.where(np.abs(mdfcounties['MDF_MaleAge'] - mdfcounties['MDF_FemaleAge']) >= 20, 1, 0)

ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'HDF 20+ Year Median Age Diff','NumCells':len(hdfcounties), 'Inconsistent':np.sum(hdfcounties['BigAgeDiff'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'MDF 20+ Year Median Age Diff','NumCells':len(mdfcounties), 'Inconsistent':np.sum(mdfcounties['BigAgeDiff'])}, index=[0])
outputdflist.append(ss)

# Counties where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women, by major race group
for r in racealonecats:
    hdfcounties_males = dfhdf[(dfhdf['RACEALONE'] == r)&(dfhdf['QSEX'] == '1')].group_by('CountyGEOID').size().reset_index(name='HDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
    mdfcounties_males = dfmdf[(dfmdf['RACEALONE'] == r)&(dfmdf['QSEX'] == '1')].group_by('CountyGEOID').size().reset_index(name='MDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
    hdfcounties_females = dfhdf[(dfhdf['RACEALONE'] == r)&(dfhdf['QSEX'] == '2')].group_by('CountyGEOID').size().reset_index(name='HDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
    mdfcounties_females = dfmdf[(dfmdf['RACEALONE'] == r)&(dfmdf['QSEX'] == '2')].group_by('CountyGEOID').size().reset_index(name='MDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])

    hdfcounties_gt5males = hdfcounties_males.loc[hdfcounties_males['HDF_Males'] >=5, 'GEOID'].tolist()
    mdfcounties_gt5males = mdfcounties_males.loc[mdfcounties_males['MDF_Males'] >=5, 'GEOID'].tolist()
    hdfcounties_gt5females = hdfcounties_females.loc[hdfcounties_females['HDF_Females'] >=5, 'GEOID'].tolist()
    mdfcounties_gt5females = mdfcounties_females.loc[mdfcounties_females['MDF_Females'] >=5, 'GEOID'].tolist()

    hdfcounties_gt5bothsex =  list(set(hdfcounties_gt5males).intersection(hdfcounties_gt5females))
    mdfcounties_gt5bothsex =  list(set(mdfcounties_gt5males).intersection(mdfcounties_gt5females))

    hdfcounties_gt5bothsex_index = pd.Index(hdfcounties_gt5bothsex, name='GEOID')
    mdfcounties_gt5bothsex_index = pd.Index(mdfcounties_gt5bothsex, name='GEOID')

    hdfcounties_malemedage = dfhdf[(dfhdf['RACEALONE'] == r)&(dfhdf['QSEX'] == '1')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(hdfcounties_gt5bothsex_index).reset_index()
    mdfcounties_malemedage = dfmdf[(dfmdf['RACEALONE'] == r)&(dfmdf['QSEX'] == '1')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(mdfcounties_gt5bothsex_index).reset_index()
    hdfcounties_femalemedage = dfhdf[(dfhdf['RACEALONE'] == r)&(dfhdf['QSEX'] == '2')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(hdfcounties_gt5bothsex_index).reset_index()
    mdfcounties_femalemedage = dfmdf[(dfmdf['RACEALONE'] == r)&(dfmdf['QSEX'] == '2')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(mdfcounties_gt5bothsex_index).reset_index()

    hdfcounties =  pd.merge(hdfcounties_malemedage, hdfcounties_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)
    mdfcounties =  pd.merge(mdfcounties_malemedage, mdfcounties_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)

    hdfcounties['BigAgeDiff'] = np.where(np.abs(hdfcounties['HDF_MaleAge'] - hdfcounties['HDF_FemaleAge']) >= 20, 1, 0)
    mdfcounties['BigAgeDiff'] = np.where(np.abs(mdfcounties['MDF_MaleAge'] - mdfcounties['MDF_FemaleAge']) >= 20, 1, 0)

    ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"HDF 20+ Year Median Age Diff {race}".format(race = racealonedict.get(r)),'NumCells':len(hdfcounties), 'Inconsistent':np.sum(hdfcounties['BigAgeDiff'])}, index=[0])
    outputdflist.append(ss)
    ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"MDF 20+ Year Median Age Diff {race}".format(race = racealonedict.get(r)),'NumCells':len(mdfcounties), 'Inconsistent':np.sum(mdfcounties['BigAgeDiff'])}, index=[0])
    outputdflist.append(ss)

# Counties where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women Hispanic
hdfcounties_males = dfhdf[(dfhdf['CENHISP'] == '2')&(dfhdf['QSEX'] == '1')].group_by('CountyGEOID').size().reset_index(name='HDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
mdfcounties_males = dfmdf[(dfmdf['CENHISP'] == '2')&(dfmdf['QSEX'] == '1')].group_by('CountyGEOID').size().reset_index(name='MDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
hdfcounties_females = dfhdf[(dfhdf['CENHISP'] == '2')&(dfhdf['QSEX'] == '2')].group_by('CountyGEOID').size().reset_index(name='HDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
mdfcounties_females = dfmdf[(dfmdf['CENHISP'] == '2')&(dfmdf['QSEX'] == '2')].group_by('CountyGEOID').size().reset_index(name='MDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])

hdfcounties_gt5males = hdfcounties_males.loc[hdfcounties_males['HDF_Males'] >=5, 'GEOID'].tolist()
mdfcounties_gt5males = mdfcounties_males.loc[mdfcounties_males['MDF_Males'] >=5, 'GEOID'].tolist()
hdfcounties_gt5females = hdfcounties_females.loc[hdfcounties_females['HDF_Females'] >=5, 'GEOID'].tolist()
mdfcounties_gt5females = mdfcounties_females.loc[mdfcounties_females['MDF_Females'] >=5, 'GEOID'].tolist()

hdfcounties_gt5bothsex =  list(set(hdfcounties_gt5males).intersection(hdfcounties_gt5females))
mdfcounties_gt5bothsex =  list(set(mdfcounties_gt5males).intersection(mdfcounties_gt5females))

hdfcounties_gt5bothsex_index = pd.Index(hdfcounties_gt5bothsex, name='GEOID')
mdfcounties_gt5bothsex_index = pd.Index(mdfcounties_gt5bothsex, name='GEOID')

hdfcounties_malemedage = dfhdf[(dfhdf['CENHISP'] == '2')&(dfhdf['QSEX'] == '1')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(hdfcounties_gt5bothsex_index).reset_index()
mdfcounties_malemedage = dfmdf[(dfmdf['CENHISP'] == '2')&(dfmdf['QSEX'] == '1')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(mdfcounties_gt5bothsex_index).reset_index()
hdfcounties_femalemedage = dfhdf[(dfhdf['CENHISP'] == '2')&(dfhdf['QSEX'] == '2')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(hdfcounties_gt5bothsex_index).reset_index()
mdfcounties_femalemedage = dfmdf[(dfmdf['CENHISP'] == '2')&(dfmdf['QSEX'] == '2')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(mdfcounties_gt5bothsex_index).reset_index()

hdfcounties =  pd.merge(hdfcounties_malemedage, hdfcounties_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)
mdfcounties =  pd.merge(mdfcounties_malemedage, mdfcounties_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)

hdfcounties['BigAgeDiff'] = np.where(np.abs(hdfcounties['HDF_MaleAge'] - hdfcounties['HDF_FemaleAge']) >= 20, 1, 0)
mdfcounties['BigAgeDiff'] = np.where(np.abs(mdfcounties['MDF_MaleAge'] - mdfcounties['MDF_FemaleAge']) >= 20, 1, 0)

ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"HDF 20+ Year Median Age Diff Hispanic",'NumCells':len(hdfcounties), 'Inconsistent':np.sum(hdfcounties['BigAgeDiff'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"MDF 20+ Year Median Age Diff Hispanic",'NumCells':len(mdfcounties), 'Inconsistent':np.sum(mdfcounties['BigAgeDiff'])}, index=[0])
outputdflist.append(ss)

# Counties where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women, Non-Hispanic White
hdfcounties_males = dfhdf[(dfhdf['RACEALONE'] == 1)&(dfhdf['CENHISP'] == '1')&(dfhdf['QSEX'] == '1')].group_by('CountyGEOID').size().reset_index(name='HDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
mdfcounties_males = dfmdf[(dfmdf['RACEALONE'] == 1)&(dfmdf['CENHISP'] == '1')&(dfmdf['QSEX'] == '1')].group_by('CountyGEOID').size().reset_index(name='MDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
hdfcounties_females = dfhdf[(dfhdf['RACEALONE'] == 1)&(dfhdf['CENHISP'] == '1')&(dfhdf['QSEX'] == '2')].group_by('CountyGEOID').size().reset_index(name='HDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])
mdfcounties_females = dfmdf[(dfmdf['RACEALONE'] == 1)&(dfmdf['CENHISP'] == '1')&(dfmdf['QSEX'] == '2')].group_by('CountyGEOID').size().reset_index(name='MDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU'])

hdfcounties_gt5males = hdfcounties_males.loc[hdfcounties_males['HDF_Males'] >=5, 'GEOID'].tolist()
mdfcounties_gt5males = mdfcounties_males.loc[mdfcounties_males['MDF_Males'] >=5, 'GEOID'].tolist()
hdfcounties_gt5females = hdfcounties_females.loc[hdfcounties_females['HDF_Females'] >=5, 'GEOID'].tolist()
mdfcounties_gt5females = mdfcounties_females.loc[mdfcounties_females['MDF_Females'] >=5, 'GEOID'].tolist()

hdfcounties_gt5bothsex =  list(set(hdfcounties_gt5males).intersection(hdfcounties_gt5females))
mdfcounties_gt5bothsex =  list(set(mdfcounties_gt5males).intersection(mdfcounties_gt5females))

hdfcounties_gt5bothsex_index = pd.Index(hdfcounties_gt5bothsex, name='GEOID')
mdfcounties_gt5bothsex_index = pd.Index(mdfcounties_gt5bothsex, name='GEOID')

hdfcounties_malemedage = dfhdf[(dfhdf['RACEALONE'] == 1)&(dfhdf['CENHISP'] == '1')&(dfhdf['QSEX'] == '1')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(hdfcounties_gt5bothsex_index).reset_index()
mdfcounties_malemedage = dfmdf[(dfmdf['RACEALONE'] == 1)&(dfmdf['CENHISP'] == '1')&(dfmdf['QSEX'] == '1')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(mdfcounties_gt5bothsex_index).reset_index()
hdfcounties_femalemedage = dfhdf[(dfhdf['RACEALONE'] == 1)&(dfhdf['CENHISP'] == '1')&(dfhdf['QSEX'] == '2')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(hdfcounties_gt5bothsex_index).reset_index()
mdfcounties_femalemedage = dfmdf[(dfmdf['RACEALONE'] == 1)&(dfmdf['CENHISP'] == '1')&(dfmdf['QSEX'] == '2')].group_by('CountyGEOID')['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU).drop(columns = ['TABBLKST', 'TABBLKCOU']).set_index('GEOID').reindex(mdfcounties_gt5bothsex_index).reset_index()

hdfcounties =  pd.merge(hdfcounties_malemedage, hdfcounties_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)
mdfcounties =  pd.merge(mdfcounties_malemedage, mdfcounties_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)

hdfcounties['BigAgeDiff'] = np.where(np.abs(hdfcounties['HDF_MaleAge'] - hdfcounties['HDF_FemaleAge']) >= 20, 1, 0)
mdfcounties['BigAgeDiff'] = np.where(np.abs(mdfcounties['MDF_MaleAge'] - mdfcounties['MDF_FemaleAge']) >= 20, 1, 0)

ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"HDF 20+ Year Median Age Diff Non-Hispanic White",'NumCells':len(hdfcounties), 'Inconsistent':np.sum(hdfcounties['BigAgeDiff'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"MDF 20+ Year Median Age Diff Non-Hispanic White",'NumCells':len(mdfcounties), 'Inconsistent':np.sum(mdfcounties['BigAgeDiff'])}, index=[0])
outputdflist.append(ss)

# Tracts where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women
hdftracts_males = dfhdf[(dfhdf['QSEX'] == '1')].group_by(['TractGEOID']).size().reset_index(name='HDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])
mdftracts_males = dfmdf[(dfmdf['QSEX'] == '1')].group_by(['TractGEOID']).size().reset_index(name='MDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])
hdftracts_females = dfhdf[(dfhdf['QSEX'] == '2')].group_by(['TractGEOID']).size().reset_index(name='HDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])
mdftracts_females = dfmdf[(dfmdf['QSEX'] == '2')].group_by(['TractGEOID']).size().reset_index(name='MDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])

hdftracts_gt5males = hdftracts_males.loc[hdftracts_males['HDF_Males'] >=5, 'GEOID'].tolist()
mdftracts_gt5males = mdftracts_males.loc[mdftracts_males['MDF_Males'] >=5, 'GEOID'].tolist()
hdftracts_gt5females = hdftracts_females.loc[hdftracts_females['HDF_Females'] >=5, 'GEOID'].tolist()
mdftracts_gt5females = mdftracts_females.loc[mdftracts_females['MDF_Females'] >=5, 'GEOID'].tolist()

hdftracts_gt5bothsex =  list(set(hdftracts_gt5males).intersection(hdftracts_gt5females))
mdftracts_gt5bothsex =  list(set(mdftracts_gt5males).intersection(mdftracts_gt5females))

hdftracts_gt5bothsex_index = pd.Index(hdftracts_gt5bothsex, name='GEOID')
mdftracts_gt5bothsex_index = pd.Index(mdftracts_gt5bothsex, name='GEOID')

hdftracts_malemedage = dfhdf[(dfhdf['QSEX'] == '1')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(hdftracts_gt5bothsex_index).reset_index()
mdftracts_malemedage = dfmdf[(dfmdf['QSEX'] == '1')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(mdftracts_gt5bothsex_index).reset_index()
hdftracts_femalemedage = dfhdf[(dfhdf['QSEX'] == '2')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(hdftracts_gt5bothsex_index).reset_index()
mdftracts_femalemedage = dfmdf[(dfmdf['QSEX'] == '2')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(mdftracts_gt5bothsex_index).reset_index()

hdftracts =  pd.merge(hdftracts_malemedage, hdftracts_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)
mdftracts =  pd.merge(mdftracts_malemedage, mdftracts_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)

hdftracts['BigAgeDiff'] = np.where(np.abs(hdftracts['HDF_MaleAge'] - hdftracts['HDF_FemaleAge']) >= 20, 1, 0)
mdftracts['BigAgeDiff'] = np.where(np.abs(mdftracts['MDF_MaleAge'] - mdftracts['MDF_FemaleAge']) >= 20, 1, 0)

ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF 20+ Year Median Age Diff','NumCells':len(hdftracts), 'Inconsistent':np.sum(hdftracts['BigAgeDiff'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF 20+ Year Median Age Diff','NumCells':len(mdftracts), 'Inconsistent':np.sum(mdftracts['BigAgeDiff'])}, index=[0])
outputdflist.append(ss)

# Tracts where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women by major race group
for r in racealonecats:
    hdftracts_males = dfhdf[(dfhdf['RACEALONE'] == r)&(dfhdf['QSEX'] == '1')].group_by(['TractGEOID']).size().reset_index(name='HDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])
    mdftracts_males = dfmdf[(dfmdf['RACEALONE'] == r)&(dfmdf['QSEX'] == '1')].group_by(['TractGEOID']).size().reset_index(name='MDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])
    hdftracts_females = dfhdf[(dfhdf['RACEALONE'] == r)&(dfhdf['QSEX'] == '2')].group_by(['TractGEOID']).size().reset_index(name='HDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])
    mdftracts_females = dfmdf[(dfmdf['RACEALONE'] == r)&(dfmdf['QSEX'] == '2')].group_by(['TractGEOID']).size().reset_index(name='MDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])

    hdftracts_gt5males = hdftracts_males.loc[hdftracts_males['HDF_Males'] >=5, 'GEOID'].tolist()
    mdftracts_gt5males = mdftracts_males.loc[mdftracts_males['MDF_Males'] >=5, 'GEOID'].tolist()
    hdftracts_gt5females = hdftracts_females.loc[hdftracts_females['HDF_Females'] >=5, 'GEOID'].tolist()
    mdftracts_gt5females = mdftracts_females.loc[mdftracts_females['MDF_Females'] >=5, 'GEOID'].tolist()

    hdftracts_gt5bothsex =  list(set(hdftracts_gt5males).intersection(hdftracts_gt5females))
    mdftracts_gt5bothsex =  list(set(mdftracts_gt5males).intersection(mdftracts_gt5females))

    hdftracts_gt5bothsex_index = pd.Index(hdftracts_gt5bothsex, name='GEOID')
    mdftracts_gt5bothsex_index = pd.Index(mdftracts_gt5bothsex, name='GEOID')

    hdftracts_malemedage = dfhdf[(dfhdf['RACEALONE'] == r)&(dfhdf['QSEX'] == '1')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(hdftracts_gt5bothsex_index).reset_index()
    mdftracts_malemedage = dfmdf[(dfmdf['RACEALONE'] == r)&(dfmdf['QSEX'] == '1')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(mdftracts_gt5bothsex_index).reset_index()
    hdftracts_femalemedage = dfhdf[(dfhdf['RACEALONE'] == r)&(dfhdf['QSEX'] == '2')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(hdftracts_gt5bothsex_index).reset_index()
    mdftracts_femalemedage = dfmdf[(dfmdf['RACEALONE'] == r)&(dfmdf['QSEX'] == '2')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(mdftracts_gt5bothsex_index).reset_index()

    hdftracts =  pd.merge(hdftracts_malemedage, hdftracts_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)
    mdftracts =  pd.merge(mdftracts_malemedage, mdftracts_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)

    hdftracts['BigAgeDiff'] = np.where(np.abs(hdftracts['HDF_MaleAge'] - hdftracts['HDF_FemaleAge']) >= 20, 1, 0)
    mdftracts['BigAgeDiff'] = np.where(np.abs(mdftracts['MDF_MaleAge'] - mdftracts['MDF_FemaleAge']) >= 20, 1, 0)

    ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':"HDF 20+ Year Median Age Diff {race}".format(race = racealonedict.get(r)),'NumCells':len(hdftracts), 'Inconsistent':np.sum(hdftracts['BigAgeDiff'])}, index=[0])
    outputdflist.append(ss)
    ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':"MDF 20+ Year Median Age Diff {race}".format(race = racealonedict.get(r)),'NumCells':len(mdftracts), 'Inconsistent':np.sum(mdftracts['BigAgeDiff'])}, index=[0])
    outputdflist.append(ss)


# Tracts where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women Hispanic
hdftracts_males = dfhdf[(dfhdf['CENHISP'] == '2')&(dfhdf['QSEX'] == '1')].group_by(['TractGEOID']).size().reset_index(name='HDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])
mdftracts_males = dfmdf[(dfmdf['CENHISP'] == '2')&(dfmdf['QSEX'] == '1')].group_by(['TractGEOID']).size().reset_index(name='MDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])
hdftracts_females = dfhdf[(dfhdf['CENHISP'] == '2')&(dfhdf['QSEX'] == '2')].group_by(['TractGEOID']).size().reset_index(name='HDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])
mdftracts_females = dfmdf[(dfmdf['CENHISP'] == '2')&(dfmdf['QSEX'] == '2')].group_by(['TractGEOID']).size().reset_index(name='MDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])

hdftracts_gt5males = hdftracts_males.loc[hdftracts_males['HDF_Males'] >=5, 'GEOID'].tolist()
mdftracts_gt5males = mdftracts_males.loc[mdftracts_males['MDF_Males'] >=5, 'GEOID'].tolist()
hdftracts_gt5females = hdftracts_females.loc[hdftracts_females['HDF_Females'] >=5, 'GEOID'].tolist()
mdftracts_gt5females = mdftracts_females.loc[mdftracts_females['MDF_Females'] >=5, 'GEOID'].tolist()

hdftracts_gt5bothsex =  list(set(hdftracts_gt5males).intersection(hdftracts_gt5females))
mdftracts_gt5bothsex =  list(set(mdftracts_gt5males).intersection(mdftracts_gt5females))

hdftracts_gt5bothsex_index = pd.Index(hdftracts_gt5bothsex, name='GEOID')
mdftracts_gt5bothsex_index = pd.Index(mdftracts_gt5bothsex, name='GEOID')

hdftracts_malemedage = dfhdf[(dfhdf['CENHISP'] == '2')&(dfhdf['QSEX'] == '1')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(hdftracts_gt5bothsex_index).reset_index()
mdftracts_malemedage = dfmdf[(dfmdf['CENHISP'] == '2')&(dfmdf['QSEX'] == '1')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(mdftracts_gt5bothsex_index).reset_index()
hdftracts_femalemedage = dfhdf[(dfhdf['CENHISP'] == '2')&(dfhdf['QSEX'] == '2')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(hdftracts_gt5bothsex_index).reset_index()
mdftracts_femalemedage = dfmdf[(dfmdf['CENHISP'] == '2')&(dfmdf['QSEX'] == '2')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(mdftracts_gt5bothsex_index).reset_index()

hdftracts =  pd.merge(hdftracts_malemedage, hdftracts_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)
mdftracts =  pd.merge(mdftracts_malemedage, mdftracts_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)

hdftracts['BigAgeDiff'] = np.where(np.abs(hdftracts['HDF_MaleAge'] - hdftracts['HDF_FemaleAge']) >= 20, 1, 0)
mdftracts['BigAgeDiff'] = np.where(np.abs(mdftracts['MDF_MaleAge'] - mdftracts['MDF_FemaleAge']) >= 20, 1, 0)

ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF 20+ Year Median Age Diff Hispanic','NumCells':len(hdftracts), 'Inconsistent':np.sum(hdftracts['BigAgeDiff'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF 20+ Year Median Age Diff Hispanic','NumCells':len(mdftracts), 'Inconsistent':np.sum(mdftracts['BigAgeDiff'])}, index=[0])
outputdflist.append(ss)

# Tracts where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women Non-Hispanic White
hdftracts_males = dfhdf[(dfhdf['RACEALONE'] == 1)&(dfhdf['CENHISP'] == '1')&(dfhdf['QSEX'] == '1')].group_by(['TractGEOID']).size().reset_index(name='HDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])
mdftracts_males = dfmdf[(dfmdf['RACEALONE'] == 1)&(dfmdf['CENHISP'] == '1')&(dfmdf['QSEX'] == '1')].group_by(['TractGEOID']).size().reset_index(name='MDF_Males').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])
hdftracts_females = dfhdf[(dfhdf['RACEALONE'] == 1)&(dfhdf['CENHISP'] == '1')&(dfhdf['QSEX'] == '2')].group_by(['TractGEOID']).size().reset_index(name='HDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])
mdftracts_females = dfmdf[(dfmdf['RACEALONE'] == 1)&(dfmdf['CENHISP'] == '1')&(dfmdf['QSEX'] == '2')].group_by(['TractGEOID']).size().reset_index(name='MDF_Females').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TABBLKST', 'TABBLKCOU','TABTRACTCE'])

hdftracts_gt5males = hdftracts_males.loc[hdftracts_males['HDF_Males'] >=5, 'GEOID'].tolist()
mdftracts_gt5males = mdftracts_males.loc[mdftracts_males['MDF_Males'] >=5, 'GEOID'].tolist()
hdftracts_gt5females = hdftracts_females.loc[hdftracts_females['HDF_Females'] >=5, 'GEOID'].tolist()
mdftracts_gt5females = mdftracts_females.loc[mdftracts_females['MDF_Females'] >=5, 'GEOID'].tolist()

hdftracts_gt5bothsex =  list(set(hdftracts_gt5males).intersection(hdftracts_gt5females))
mdftracts_gt5bothsex =  list(set(mdftracts_gt5males).intersection(mdftracts_gt5females))

hdftracts_gt5bothsex_index = pd.Index(hdftracts_gt5bothsex, name='GEOID')
mdftracts_gt5bothsex_index = pd.Index(mdftracts_gt5bothsex, name='GEOID')

hdftracts_malemedage = dfhdf[(dfhdf['RACEALONE'] == 1)&(dfhdf['CENHISP'] == '1')&(dfhdf['QSEX'] == '1')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(hdftracts_gt5bothsex_index).reset_index()
mdftracts_malemedage = dfmdf[(dfmdf['RACEALONE'] == 1)&(dfmdf['CENHISP'] == '1')&(dfmdf['QSEX'] == '1')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_MaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(mdftracts_gt5bothsex_index).reset_index()
hdftracts_femalemedage = dfhdf[(dfhdf['RACEALONE'] == 1)&(dfhdf['CENHISP'] == '1')&(dfhdf['QSEX'] == '2')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='HDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(hdftracts_gt5bothsex_index).reset_index()
mdftracts_femalemedage = dfmdf[(dfmdf['RACEALONE'] == 1)&(dfmdf['CENHISP'] == '1')&(dfmdf['QSEX'] == '2')].group_by(['TractGEOID'])['QAGE'].aggregate(lambda x: median_grouped(x+0.5)).reset_index(name='MDF_FemaleAge').assign(GEOID = lambda x: x.TABBLKST + x.TABBLKCOU + x.TABTRACTCE).drop(columns = ['TractGEOID']).set_index('GEOID').reindex(mdftracts_gt5bothsex_index).reset_index()

hdftracts =  pd.merge(hdftracts_malemedage, hdftracts_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)
mdftracts =  pd.merge(mdftracts_malemedage, mdftracts_femalemedage, on='GEOID', how = 'inner', validate = mergeValidation)

hdftracts['BigAgeDiff'] = np.where(np.abs(hdftracts['HDF_MaleAge'] - hdftracts['HDF_FemaleAge']) >= 20, 1, 0)
mdftracts['BigAgeDiff'] = np.where(np.abs(mdftracts['MDF_MaleAge'] - mdftracts['MDF_FemaleAge']) >= 20, 1, 0)

ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF 20+ Year Median Age Diff Non-Hispanic White','NumCells':len(hdftracts), 'Inconsistent':np.sum(hdftracts['BigAgeDiff'])}, index=[0])
outputdflist.append(ss)
ss = pd.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF 20+ Year Median Age Diff Non-Hispanic White','NumCells':len(mdftracts), 'Inconsistent':np.sum(mdftracts['BigAgeDiff'])}, index=[0])
outputdflist.append(ss)

# Output
outputdf = pd.concat(outputdflist, ignore_index=True)
outputdf.to_csv(f"{OUTPUTDIR}/cef_per_metrics_v2.csv", index=False)
print("{} All Done".format(datetime.now()))
