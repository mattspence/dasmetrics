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
        mean_hdf = data['HDF_Population'].drop_nulls().mean()
        rmse_val = (data['Diff'].pow(2).mean()) ** 0.5
        odf = pl.DataFrame({
            'Geography': [geography],
            'Size_Category': [sizecategory],
            'Characteristic': [characteristic],
            'MinDiff': [data['Diff'].drop_nulls().min()],
            'MeanDiff': [data['Diff'].drop_nulls().mean()],
            'MedianDiff': [data['Diff'].drop_nulls().median()],
            'MaxDiff': [data['Diff'].drop_nulls().max()],
            'MeanAbsDiff': [data['AbsDiff'].drop_nulls().mean()],
            'MedianAbsDiff': [data['AbsDiff'].drop_nulls().median()],
            'AbsDiff90th': [data['AbsDiff'].drop_nulls().quantile(0.90, interpolation='linear')],
            'AbsDiff95th': [data['AbsDiff'].drop_nulls().quantile(0.95, interpolation='linear')],
            'MinPercDiff': [data['PercDiff'].drop_nulls().min()],
            'MeanPercDiff': [data['PercDiff'].drop_nulls().mean()],
            'MedianPercDiff': [data['PercDiff'].drop_nulls().median()],
            'MaxPercDiff': [data['PercDiff'].drop_nulls().max()],
            'PercDiffNAs': [data['PercDiff'].is_null().sum()],
            'MeanAbsPercDiff': [data['AbsPercDiff'].drop_nulls().mean()],
            'MedianAbsPercDiff': [data['AbsPercDiff'].drop_nulls().median()],
            'AbsPercDiff90th': [data['AbsPercDiff'].drop_nulls().quantile(0.90, interpolation='linear')],
            'AbsPercDiff95th': [data['AbsPercDiff'].drop_nulls().quantile(0.95, interpolation='linear')],
            'AbsPercDiffMax': [data['AbsPercDiff'].drop_nulls().max()],
            'AbsPercDiffNAs': [data['AbsPercDiff'].is_null().sum()],
            'RMSE': [rmse_val],
            'CV': [100 * (rmse_val / mean_hdf) if mean_hdf is not None and mean_hdf != 0 else None],
            'MeanCEFPop': [mean_hdf],
            'NumCells': [len(data)],
            'NumberBtw2Perc5Perc': [len(data.filter((pl.col('AbsPercDiff') >= 2) & (pl.col('AbsPercDiff') <= 5)))],
            'NumberGreater5Perc': [len(data.filter(pl.col('AbsPercDiff') > 5))],
            'NumberGreater200': [len(data.filter(pl.col('AbsDiff') > 200))],
            'NumberGreater10Perc': [len(data.filter(pl.col('AbsPercDiff') > 10))],
        })
    else:
        odf = pl.DataFrame({
            'Geography': [geography],
            'Size_Category': [sizecategory],
            'Characteristic': [characteristic],
            'MinDiff': [0.0], 'MeanDiff': [0.0], 'MedianDiff': [0.0], 'MaxDiff': [0.0],
            'MeanAbsDiff': [0.0], 'MedianAbsDiff': [0.0], 'AbsDiff90th': [0.0], 'AbsDiff95th': [0.0],
            'MinPercDiff': [0.0], 'MeanPercDiff': [0.0], 'MedianPercDiff': [0.0], 'MaxPercDiff': [0.0],
            'PercDiffNAs': [0], 'MeanAbsPercDiff': [0.0], 'MedianAbsPercDiff': [0.0],
            'AbsPercDiff90th': [0.0], 'AbsPercDiff95th': [0.0], 'AbsPercDiffMax': [0.0], 'AbsPercDiffNAs': [0],
            'RMSE': [0.0], 'CV': [0.0], 'MeanCEFPop': [0.0], 'NumCells': [0],
            'NumberBtw2Perc5Perc': [0], 'NumberGreater5Perc': [0], 'NumberGreater200': [0], 'NumberGreater10Perc': [0],
        })
    return odf


def dfhdf():
    return pl.scan_ipc('hdf.arrow')

def dfmdf():
    return pl.scan_ipc('mdf.arrow')

# Counties Total Population
hdfcounties_totalpop = dfhdf().group_by('CountyGEOID').agg(HDF_Population = pl.len())
mdfcounties_totalpop = dfmdf().group_by('CountyGEOID').agg(MDF_Population = pl.len())
counties_totalpop =  hdfcounties_totalpop.join(mdfcounties_totalpop, on='CountyGEOID', how='full', coalesce = True).pipe(calculate_stats).collect()
counties_totalpop.write_csv(f"{OUTPUTDIR}/counties_totalpop.csv")
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
places_totalpop.write_csv(f"{OUTPUTDIR}/places_totalpop.csv")
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
tracts_totalpop.write_csv(f"{OUTPUTDIR}/tracts_totalpop.csv")
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
elemschdists_totalpop.write_csv(f"{OUTPUTDIR}/elemschdists_totalpop.csv")
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
secschdists_totalpop.write_csv(f"{OUTPUTDIR}/secschdists_totalpop.csv")
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
unischdists_totalpop.write_csv(f"{OUTPUTDIR}/unischdists_totalpop.csv")
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
mcds_totalpop.write_csv(f"{OUTPUTDIR}/mcds_totalpop.csv")
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
fedairs_totalpop.write_csv(f"{OUTPUTDIR}/fedairs_totalpop.csv")
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
otsas_totalpop.write_csv(f"{OUTPUTDIR}/otsas_totalpop.csv")
ss = otsas_totalpop.pipe(calculate_ss, geography="OTSA", sizecategory = "All", characteristic = "Total Population")
outputdflist.append(ss)

# ANVSA Total Population
hdfanvsas_totalpop = dfhdf().group_by(['ANVSAGEOID']).agg(HDF_Population = pl.len())
mdfanvsas_totalpop = dfmdf().group_by(['ANVSAGEOID']).agg(MDF_Population = pl.len())
anvsas_totalpop = allanvsasdf.join(hdfanvsas_totalpop.join(mdfanvsas_totalpop, on='ANVSAGEOID', how='full', coalesce=True), how='left', on='ANVSAGEOID').pipe(calculate_stats).collect()
anvsas_totalpop.write_csv(f"{OUTPUTDIR}/anvsas_totalpop.csv")
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
aiannh_totalpop.write_csv(f"{OUTPUTDIR}/aiannh_totalpop.csv")
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
hdftracts_totalpop18p = dfhdf().filter(pl.col('QAGE') >= 18).group_by('TractGEOID').agg(HDF_Population = pl.len())
mdftracts_totalpop18p = dfmdf().filter(pl.col('QAGE') >= 18).group_by('TractGEOID').agg(MDF_Population = pl.len())
tracts_totalpop18p =  alltractsdf.join(hdftracts_totalpop18p.join(mdftracts_totalpop18p, on='TractGEOID', how='full', coalesce=True), on='TractGEOID', how='left').pipe(calculate_stats).collect()
ss = tracts_totalpop18p.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Total Population Aged 18+")
outputdflist.append(ss)

# TODO
# if runPRhere:
#     # PR Counties/Municipios Total Population 18+
#     hdfcountiespr_totalpop18p = dfhdfpr().filter(pl.col('QAGE') >= 18).group_by('CountyGEOID').agg(HDF_Population = pl.len())
#     mdfcountiespr_totalpop18p = dfmdfpr().filter(pl.col('QAGE') >= 18).group_by('CountyGEOID').agg(MDF_Population = pl.len())
#     countiespr_totalpop18p =  hdfcountiespr_totalpop18p.join(mdfcountiespr_totalpop18p, on='GEOID', how="full", coalesce=True).pipe(calculate_stats)
#     ss = countiespr_totalpop18p.pipe(calculate_ss, geography="PR County/Municipio", sizecategory = "All", characteristic = "Total Population Aged 18+")
#     outputdflist.append(ss)

#     # PR Tracts Total Population 18+
#     hdftractspr_totalpop18p = dfhdfpr().filter(pl.col('QAGE') >= 18).group_by('TractGEOID').agg(HDF_Population = pl.len())
#     mdftractspr_totalpop18p = dfmdfpr().filter(pl.col('QAGE') >= 18).group_by('TractGEOID').agg(MDF_Population = pl.len())
#     tractspr_totalpop18p =  hdftractspr_totalpop18p.join(mdftractspr_totalpop18p, on='GEOID', how="full", coalesce=True).pipe(calculate_stats)
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
        # temp = counties_racealone.filter(pl.col('Race_PopSize') == i)
        # temp.write_csv(f"{OUTPUTDIR}/counties_racealone_race{r}_{i}.csv")
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
        # temp = places_racealone.filter(pl.col('Race_PopSize') == i)
        # temp.write_csv(f"{OUTPUTDIR}/places_racealone_race{r}_{i}.csv")
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
    hdfstates_hispracealone = dfhdf().filter((pl.col('CENHISP') == '2'),(pl.col('RACEALONE') == r)).group_by('StateGEOID').agg(HDF_Population = pl.len())
    mdfstates_hispracealone = dfmdf().filter((pl.col('CENHISP') == '2'),(pl.col('RACEALONE') == r)).group_by('StateGEOID').agg(MDF_Population = pl.len())
    states_hispracealone =  allstatesdf.join(hdfstates_hispracealone.join(mdfstates_hispracealone, on='StateGEOID', how='full', coalesce=True), how='left', on='StateGEOID').pipe(calculate_stats).collect()
    ss = states_hispracealone.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "Hispanic {race}".format(race = racealonedict.get(r)))
    outputdflist.append(ss)
    hdfstates_nonhispracealone = dfhdf().filter((pl.col('CENHISP') == '1'),(pl.col('RACEALONE') == r)).group_by('StateGEOID').agg(HDF_Population = pl.len())
    mdfstates_nonhispracealone = dfmdf().filter((pl.col('CENHISP') == '1'),(pl.col('RACEALONE') == r)).group_by('StateGEOID').agg(MDF_Population = pl.len())
    states_nonhispracealone =  allstatesdf.join(hdfstates_nonhispracealone.join(mdfstates_nonhispracealone, on='StateGEOID', how='full', coalesce=True), how='left', on='StateGEOID').pipe(calculate_stats).collect()
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
        ss = counties_hispracealone.filter(pl.col('HispRace_PopSize') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Hispanic {race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)
    counties_nonhispracealone = counties_nonhispracealone.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in counties_nonhispracealone['HispRace_PopSize'].cat.get_categories():
        ss = counties_nonhispracealone.filter(pl.col('HispRace_PopSize') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Non-Hispanic {race}".format(race = racealonedict.get(r)))
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
        ss = places_hispracealone.filter(pl.col('HispRace_PopSize') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "Hispanic {race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)
    places_nonhispracealone = places_nonhispracealone.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in places_nonhispracealone['HispRace_PopSize'].cat.get_categories():
        ss = places_nonhispracealone.filter(pl.col('HispRace_PopSize') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "Non-Hispanic {race}".format(race = racealonedict.get(r)))
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
        ss = tracts_hispracealone.filter(pl.col('HispRace_PopSize') == i).pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Hispanic {race}".format(race = racealonedict.get(r)))
        outputdflist.append(ss)
    tracts_nonhispracealone = tracts_nonhispracealone.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in tracts_nonhispracealone['HispRace_PopSize'].cat.get_categories():
        ss = tracts_nonhispracealone.filter(pl.col('HispRace_PopSize') == i).pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Non-Hispanic {race}".format(race = racealonedict.get(r)))
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
        ss = tracts_hispracealone18p.filter(pl.col('HispRace_PopSize') == i).pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Hispanic {race} Aged 18+".format(race = racealonedict.get(r)))
        outputdflist.append(ss)
    tracts_nonhispracealone18p = tracts_nonhispracealone18p.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in tracts_nonhispracealone18p['HispRace_PopSize'].cat.get_categories():
        ss = tracts_nonhispracealone18p.filter(pl.col('HispRace_PopSize') == i).pipe(calculate_ss, geography="Tract", sizecategory = str(i), characteristic = "Non-Hispanic {race} Aged 18+".format(race = racealonedict.get(r)))
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
        ss = blockgroups_hispracealone18p.filter(pl.col('HispRace_PopSize') == i).pipe(calculate_ss, geography="Block Group", sizecategory = str(i), characteristic = "Hispanic {race} Aged 18+".format(race = racealonedict.get(r)))
        outputdflist.append(ss)
    blockgroups_nonhispracealone18p = blockgroups_nonhispracealone18p.with_columns(HispRace_PopSize = pl.col('HDF_Population').cut(breaks= [10,100], left_closed = True))
    for i in blockgroups_nonhispracealone18p['HispRace_PopSize'].cat.get_categories():
        ss = blockgroups_nonhispracealone18p.filter(pl.col('HispRace_PopSize') == i).pipe(calculate_ss, geography="Block Group", sizecategory = str(i), characteristic = "Non-Hispanic {race} Aged 18+".format(race = racealonedict.get(r)))
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
    hdfcounties_3gage = dfhdf().filter(pl.col('QAGE_3G') == g).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_3gage = dfmdf().filter(pl.col('QAGE_3G') == g).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_3gage =  hdfcounties_3gage.join(mdfcounties_3gage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = counties_3gage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Age {age}".format(age=g))
    outputdflist.append(ss)
for s in sexcats:
    hdfcounties_sex = dfhdf().filter(pl.col('QSEX') == s).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_sex = dfmdf().filter(pl.col('QSEX') == s).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_sex =  hdfcounties_sex.join(mdfcounties_sex, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = counties_sex.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
for s in sexcats:
    for g in qage_3g_cats:
        hdfcounties_sex3gage = dfhdf().filter((pl.col('QAGE_3G') == g), (pl.col('QSEX') == s).group_by('CountyGEOID').agg(HDF_Population = pl.len())
        mdfcounties_sex3gage = dfmdf().filter((pl.col('QAGE_3G') == g), (pl.col('QSEX') == s)).group_by('CountyGEOID').agg(MDF_Population = pl.len())
        counties_sex3gage =  hdfcounties_sex3gage.join(mdfcounties_sex3gage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = counties_sex3gage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

print("{} County Sex by 3 Age Done".format(datetime.now()))

# County Sex by 3 Age Groups [0.0, 1000.0)]
# Must use .reindex(counties_lt1000index, fill_value=0).reset_index()
for g in qage_3g_cats:
    hdfcounties_3gage = dfhdf().filter(pl.col('QAGE_3G') == g).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_3gage = dfmdf().filter(pl.col('QAGE_3G') == g).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_3gage =  hdfcounties_3gage.join(mdfcounties_3gage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = counties_3gage.pipe(calculate_ss, geography="County", sizecategory = "[0.0, 1000.0)", characteristic = "Age {age}".format(age=g))
    outputdflist.append(ss)
for s in sexcats:
    hdfcounties_sex = dfhdf().filter(pl.col('QSEX') == s).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_sex = dfmdf().filter(pl.col('QSEX') == s).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_sex =  hdfcounties_sex.join(mdfcounties_sex, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = counties_sex.pipe(calculate_ss, geography="County", sizecategory = "[0.0, 1000.0)", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
for s in sexcats:
    for g in qage_3g_cats:
        hdfcounties_sex3gage = dfhdf().filter((pl.col('QAGE_3G') == g), (pl.col('QSEX') == s).group_by('CountyGEOID').agg(HDF_Population = pl.len())
        mdfcounties_sex3gage = dfmdf().filter((pl.col('QAGE_3G') == g), (pl.col('QSEX') == s)).group_by('CountyGEOID').agg(MDF_Population = pl.len())
        counties_sex3gage =  hdfcounties_sex3gage.join(mdfcounties_sex3gage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = counties_sex3gage.pipe(calculate_ss, geography="County", sizecategory = "[0.0, 1000.0)", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

# Place Sex by 3 Age Groups
for g in qage_3g_cats:
    hdfplaces_3gage = dfhdf().filter(pl.col('QAGE_3G') == g).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_3gage = dfmdf().filter(pl.col('QAGE_3G') == g).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_3gage =  hdfplaces_3gage.join(mdfplaces_3gage, on=['IncPlaceGEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = places_3gage.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
for s in sexcats:
    hdfplaces_sex = dfhdf().filter(pl.col('QSEX') == s).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_sex = dfmdf().filter(pl.col('QSEX') == s).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_sex =  hdfplaces_sex.join(mdfplaces_sex, on=['IncPlaceGEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = places_sex.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
for s in sexcats:
    for g in qage_3g_cats:
        hdfplaces_sex3gage = dfhdf().filter((pl.col('QAGE_3G') == g) & (pl.col('QSEX')== s).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
        mdfplaces_sex3gage = dfmdf().filter((pl.col('QAGE_3G') == g) & (pl.col('QSEX')== s).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
        places_sex3gage =  hdfplaces_sex3gage.join(mdfplaces_sex3gage, on=['IncPlaceGEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = places_sex3gage.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

# Place Sex by 3 Age Groups [0.0, 500.0)
# Must use .reindex(places_lt500index, fill_value=0).reset_index()
for g in qage_3g_cats:
    hdfplaces_3gage = dfhdf().filter(pl.col('QAGE_3G') == g).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_3gage = dfmdf().filter(pl.col('QAGE_3G') == g).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_3gage =  hdfplaces_3gage.join(mdfplaces_3gage, on=['IncPlaceGEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = places_3gage.pipe(calculate_ss, geography="Place", sizecategory = "[0.0, 500.0)", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
for s in sexcats:
    hdfplaces_sex = dfhdf().filter(pl.col('QSEX') == s).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_sex = dfmdf().filter(pl.col('QSEX') == s).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_sex =  hdfplaces_sex.join(mdfplaces_sex, on=['IncPlaceGEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = places_sex.pipe(calculate_ss, geography="Place", sizecategory = "[0.0, 500.0)", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
for s in sexcats:
    for g in qage_3g_cats:
        hdfplaces_sex3gage = dfhdf().filter((pl.col('QAGE_3G') == g) & (pl.col('QSEX')== s).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
        mdfplaces_sex3gage = dfmdf().filter((pl.col('QAGE_3G') == g) & (pl.col('QSEX')== s).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
        places_sex3gage =  hdfplaces_sex3gage.join(mdfplaces_sex3gage, on=['IncPlaceGEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = places_sex3gage.pipe(calculate_ss, geography="Place", sizecategory = "[0.0, 500.0)", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

# Tract Sex by 3 Age Groups
for g in qage_3g_cats:
    hdftracts_3gage = dfhdf().filter(pl.col('QAGE_3G') == g).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
    mdftracts_3gage = dfmdf().filter(pl.col('QAGE_3G') == g).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
    tracts_3gage =  hdftracts_3gage.join(mdftracts_3gage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = tracts_3gage.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
for s in sexcats:
    hdftracts_sex = dfhdf().filter(pl.col('QSEX') == s).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
    mdftracts_sex = dfmdf().filter(pl.col('QSEX') == s).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
    tracts_sex =  hdftracts_sex.join(mdftracts_sex, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = tracts_sex.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
for s in sexcats:
    for g in qage_3g_cats:
        hdftracts_sex3gage = dfhdf().filter((pl.col('QAGE_3G') == g) & (pl.col('QSEX')== s).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
        mdftracts_sex3gage = dfmdf().filter((pl.col('QAGE_3G') == g) & (pl.col('QSEX')== s).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
        tracts_sex3gage =  hdftracts_sex3gage.join(mdftracts_sex3gage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = tracts_sex3gage.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)
print("{} Sex By 3 Age Groups Done".format(datetime.now()))

# County 5-Year Age Groups
for g in qage_5y_cats:
    hdfcounties_5yage = dfhdf().filter(pl.col('QAGE_5Y') == g).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_5yage = dfmdf().filter(pl.col('QAGE_5Y') == g).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_5yage =  hdfcounties_5yage.join(mdfcounties_5yage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = counties_5yage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
# County Sex x 5-Year Age Groups
for s in sexcats:
    for g in qage_5y_cats:
        hdfcounties_sex5yage = dfhdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by('CountyGEOID').agg(HDF_Population = pl.len())
        mdfcounties_sex5yage = dfmdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by('CountyGEOID').agg(MDF_Population = pl.len())
        counties_sex5yage =  hdfcounties_sex5yage.join(mdfcounties_sex5yage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = counties_sex5yage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

if runAGEBYRACE:
    # County 5-Year Age Groups by RACEALONE
    for r in racealonecats:
        # for g in qage_5y_cats:
        #     hdfcounties_5yage = dfhdf().filter((pl.col('QAGE_5Y') == g)&(pl.col('RACEALONE') == r).group_by('CountyGEOID').agg(HDF_Population = pl.len())
        #     mdfcounties_5yage = dfmdf().filter((pl.col('QAGE_5Y') == g)&(pl.col('RACEALONE') == r).group_by('CountyGEOID').agg(MDF_Population = pl.len())
        #     counties_5yage =  hdfcounties_5yage.join(mdfcounties_5yage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
        #     ss = counties_5yage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{race} Age {agegroup}".format(race = racealonedict.get(r),agegroup=g))
        #     outputdflist.append(ss)
        for s in sexcats:
            for g in qage_5y_cats:
                hdfcounties_sex5yage = dfhdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s)&(pl.col('RACEALONE') == r).group_by('CountyGEOID').agg(HDF_Population = pl.len())
                mdfcounties_sex5yage = dfmdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s)&(pl.col('RACEALONE') == r).group_by('CountyGEOID').agg(MDF_Population = pl.len())
                counties_sex5yage =  hdfcounties_sex5yage.join(mdfcounties_sex5yage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
                ss = counties_sex5yage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{race} {sex} Age {agegroup}".format(race = racealonedict.get(r), sex = sexdict.get(s), agegroup = g))
                outputdflist.append(ss)

    # County 5-Year Age Groups by RACE AOIC
    for rg in racegroups:
        # for g in qage_5y_cats:
        #     hdfcounties_5yage = dfhdf().filter((pl.col('QAGE_5Y') == g)&(dfhdf[rg]==1).group_by('CountyGEOID').agg(HDF_Population = pl.len())
        #     mdfcounties_5yage = dfmdf().filter((pl.col('QAGE_5Y') == g)&(dfmdf[rg]==1).group_by('CountyGEOID').agg(MDF_Population = pl.len())
        #     counties_5yage =  hdfcounties_5yage.join(mdfcounties_5yage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
        #     ss = counties_5yage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{race} Age {agegroup}".format(race = raceincombdict.get(rg),agegroup=g))
        #     outputdflist.append(ss)
        for s in sexcats:
            for g in qage_5y_cats:
                hdfcounties_sex5yage = dfhdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s)&(dfhdf[rg]==1).group_by('CountyGEOID').agg(HDF_Population = pl.len())
                mdfcounties_sex5yage = dfmdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s)&(dfmdf[rg]==1).group_by('CountyGEOID').agg(MDF_Population = pl.len())
                counties_sex5yage =  hdfcounties_sex5yage.join(mdfcounties_sex5yage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
                ss = counties_sex5yage.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "{race} {sex} Age {agegroup}".format(race = raceincombdict.get(rg), sex = sexdict.get(s), agegroup = g))
                outputdflist.append(ss)

# Place 5-Year Age Groups
for g in qage_5y_cats:
    hdfplaces_5yage = dfhdf().filter(pl.col('QAGE_5Y') == g).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_5yage = dfmdf().filter(pl.col('QAGE_5Y') == g).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_5yage =  hdfplaces_5yage.join(mdfplaces_5yage, on=['IncPlaceGEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = places_5yage.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
# Place Sex x 5-Year Age Groups
for s in sexcats:
    for g in qage_5y_cats:
        hdfplaces_sex5yage = dfhdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
        mdfplaces_sex5yage = dfmdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
        places_sex5yage =  hdfplaces_sex5yage.join(mdfplaces_sex5yage, on=['IncPlaceGEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = places_sex5yage.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)


# Tract 5-Year Age Groups
for g in qage_5y_cats:
    hdftracts_5yage = dfhdf().filter(pl.col('QAGE_5Y') == g).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
    mdftracts_5yage = dfmdf().filter(pl.col('QAGE_5Y') == g).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
    tracts_5yage =  hdftracts_5yage.join(mdftracts_5yage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = tracts_5yage.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
# Tract Sex x 5-Year Age Groups
for s in sexcats:
    for g in qage_5y_cats:
        hdftracts_sex5yage = dfhdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
        mdftracts_sex5yage = dfmdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
        tracts_sex5yage =  hdftracts_sex5yage.join(mdftracts_sex5yage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = tracts_sex5yage.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

if runPRhere:
    # PR Counties/Municipios Sex and 5-Year Age Groups
    for s in sexcats:
        hdfcountiespr_sex = dfhdfpr().filter(pl.col('QSEX') == s).group_by('CountyGEOID').agg(HDF_Population = pl.len())
        mdfcountiespr_sex = dfmdfpr().filter(pl.col('QSEX') == s).group_by('CountyGEOID').agg(MDF_Population = pl.len())
        countiespr_sex =  hdfcountiespr_sex.join(mdfcountiespr_sex, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = countiespr_sex.pipe(calculate_ss, geography="PR County/Municipio", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
        outputdflist.append(ss)
    # PR Counties/Municipios 5-Year Age Groups
    for g in qage_5y_cats:
        hdfcountiespr_5yage = dfhdfpr().filter(pl.col('QAGE_5Y') == g).group_by('CountyGEOID').agg(HDF_Population = pl.len())
        mdfcountiespr_5yage = dfmdfpr().filter(pl.col('QAGE_5Y') == g).group_by('CountyGEOID').agg(MDF_Population = pl.len())
        countiespr_5yage =  hdfcountiespr_5yage.join(mdfcountiespr_5yage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = countiespr_5yage.pipe(calculate_ss, geography="PR County/Municipio", sizecategory = "All", characteristic = "Age {}".format(g))
        outputdflist.append(ss)
    # PR Counties/Municipios Sex x 5-Year Age Groups
    for s in sexcats:
        for g in qage_5y_cats:
            hdfcountiespr_sex5yage = dfhdfpr().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by('CountyGEOID').agg(HDF_Population = pl.len())
            mdfcountiespr_sex5yage = dfmdfpr().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by('CountyGEOID').agg(MDF_Population = pl.len())
            countiespr_sex5yage =  hdfcountiespr_sex5yage.join(mdfcountiespr_sex5yage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
            ss = countiespr_sex5yage.pipe(calculate_ss, geography="PR County/Municipio", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
            outputdflist.append(ss)
    # PR Tract Sex and 5-Year Age Groups
    for s in sexcats:
        hdftractspr_sex = dfhdfpr().filter(pl.col('QSEX') == s).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
        mdftractspr_sex = dfmdfpr().filter(pl.col('QSEX') == s).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
        tractspr_sex =  hdftractspr_sex.join(mdftractspr_sex, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = tractspr_sex.pipe(calculate_ss, geography="PR Tract", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
        outputdflist.append(ss)
    # PR Tract 5-Year Age Groups
    for g in qage_5y_cats:
        hdftractspr_5yage = dfhdfpr().filter(pl.col('QAGE_5Y') == g).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
        mdftractspr_5yage = dfmdfpr().filter(pl.col('QAGE_5Y') == g).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
        tractspr_5yage =  hdftractspr_5yage.join(mdftractspr_5yage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = tractspr_5yage.pipe(calculate_ss, geography="PR Tract", sizecategory = "All", characteristic = "Age {}".format(g))
        outputdflist.append(ss)
    # PR Tract Sex x 5-Year Age Groups
    for s in sexcats:
        for g in qage_5y_cats:
            hdftractspr_sex5yage = dfhdfpr().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
            mdftractspr_sex5yage = dfmdfpr().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
            tractspr_sex5yage =  hdftractspr_sex5yage.join(mdftractspr_sex5yage, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
            ss = tractspr_sex5yage.pipe(calculate_ss, geography="PR Tract", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
            outputdflist.append(ss)

# Federal AIR Sex and 5-Year Age Groups
for s in sexcats:
    hdffedairs_sex = dfhdf().filter(pl.col('QSEX') == s).group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
    mdffedairs_sex = dfmdf().filter(pl.col('QSEX') == s).group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
    fedairs_sex =  hdffedairs_sex.join(mdffedairs_sex, on=['FedAIRGEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = fedairs_sex.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
# Federal AIR 5-Year Age Groups
for g in qage_5y_cats:
    hdffedairs_5yage = dfhdf().filter(pl.col('QAGE_5Y') == g).group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
    mdffedairs_5yage = dfmdf().filter(pl.col('QAGE_5Y') == g).group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
    fedairs_5yage =  hdffedairs_5yage.join(mdffedairs_5yage, on=['FedAIRGEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = fedairs_5yage.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
#    Federal AIR Sex x 5-Year Age Groups
for s in sexcats:
    for g in qage_5y_cats:
        hdffedairs_sex5yage = dfhdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
        mdffedairs_sex5yage = dfmdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
        fedairs_sex5yage =  hdffedairs_sex5yage.join(mdffedairs_sex5yage, on=['FedAIRGEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = fedairs_sex5yage.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

if runAGEBYRACE:
    # Federal AIR 5-Year Age Groups by RACE ALONE
    for r in racealonecats:
        for s in sexcats:
            for g in qage_5y_cats:
                hdffedairs_sex5yage = dfhdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s)&(pl.col('RACEALONE') == r).group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
                mdffedairs_sex5yage = dfmdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s)&(pl.col('RACEALONE') == r).group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
                fedairs_sex5yage =  hdffedairs_sex5yage.join(mdffedairs_sex5yage, on=['FedAIRGEOID'], how="full", coalesce=True).pipe(calculate_stats)
                ss = fedairs_sex5yage.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "{race} {sex} Age {agegroup}".format(race = racealonedict.get(r), sex = sexdict.get(s), agegroup = g))
                outputdflist.append(ss)
    # Fed AIR 5-Year Age Groups by RACE AOIC
    for rg in racegroups:
        for s in sexcats:
            for g in qage_5y_cats:
                hdffedairs_sex5yage = dfhdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s) & (dfhdf[rg]==1).group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
                mdffedairs_sex5yage = dfmdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s) & (dfmdf[rg]==1).group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
                fedairs_sex5yage =  hdffedairs_sex5yage.join(mdffedairs_sex5yage, on=['FedAIRGEOID'], how="full", coalesce=True).pipe(calculate_stats)
                ss = fedairs_sex5yage.pipe(calculate_ss, geography="Fed AIR", sizecategory = "All", characteristic = "{race} {sex} Age {agegroup}".format(race = raceincombdict.get(rg), sex = sexdict.get(s), agegroup = g))
                outputdflist.append(ss)


# OTSA Sex and 5-Year Age Groups
for s in sexcats:
    hdfotsas_sex = dfhdf().filter(pl.col('QSEX') == s).group_by(['OTSAGEOID']).agg(HDF_Population = pl.len())
    mdfotsas_sex = dfmdf().filter(pl.col('QSEX') == s).group_by(['OTSAGEOID']).agg(MDF_Population = pl.len())
    otsas_sex =  hdfotsas_sex.join(mdfotsas_sex, on=['OTSAGEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = otsas_sex.pipe(calculate_ss, geography="OTSA", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
# OTSA 5-Year Age Groups
for g in qage_5y_cats:
    hdfotsas_5yage = dfhdf().filter(pl.col('QAGE_5Y') == g).group_by(['OTSAGEOID']).agg(HDF_Population = pl.len())
    mdfotsas_5yage = dfmdf().filter(pl.col('QAGE_5Y') == g).group_by(['OTSAGEOID']).agg(MDF_Population = pl.len())
    otsas_5yage =  hdfotsas_5yage.join(mdfotsas_5yage, on=['OTSAGEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = otsas_5yage.pipe(calculate_ss, geography="OTSA", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
# OTSA Sex x 5-Year Age Groups
for s in sexcats:
    for g in qage_5y_cats:
        hdfotsas_sex5yage = dfhdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by(['OTSAGEOID']).agg(HDF_Population = pl.len())
        mdfotsas_sex5yage = dfmdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by(['OTSAGEOID']).agg(MDF_Population = pl.len())
        otsas_sex5yage =  hdfotsas_sex5yage.join(mdfotsas_sex5yage, on=['OTSAGEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = otsas_sex5yage.pipe(calculate_ss, geography="OTSA", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

# ANVSA Sex and 5-Year Age Groups
for s in sexcats:
    hdfanvsas_sex = dfhdf().filter(pl.col('QSEX') == s).group_by(['ANVSAGEOID']).agg(HDF_Population = pl.len())
    mdfanvsas_sex = dfmdf().filter(pl.col('QSEX') == s).group_by(['ANVSAGEOID']).agg(MDF_Population = pl.len())
    anvsas_sex =  hdfanvsas_sex.join(mdfanvsas_sex, on=['ANVSAGEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = anvsas_sex.pipe(calculate_ss, geography="ANVSA", sizecategory = "All", characteristic = "{sex}".format(sex = sexdict.get(s)))
    outputdflist.append(ss)
# ANVSA 5-Year Age Groups
for g in qage_5y_cats:
    hdfanvsas_5yage = dfhdf().filter(pl.col('QAGE_5Y') == g).group_by(['ANVSAGEOID']).agg(HDF_Population = pl.len())
    mdfanvsas_5yage = dfmdf().filter(pl.col('QAGE_5Y') == g).group_by(['ANVSAGEOID']).agg(MDF_Population = pl.len())
    anvsas_5yage =  hdfanvsas_5yage.join(mdfanvsas_5yage, on=['ANVSAGEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = anvsas_5yage.pipe(calculate_ss, geography="ANVSA", sizecategory = "All", characteristic = "Age {}".format(g))
    outputdflist.append(ss)
# ANVSA Sex x 5-Year Age Groups
for s in sexcats:
    for g in qage_5y_cats:
        hdfanvsas_sex5yage = dfhdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by(['ANVSAGEOID']).agg(HDF_Population = pl.len())
        mdfanvsas_sex5yage = dfmdf().filter((pl.col('QAGE_5Y') == g) & (pl.col('QSEX')== s).group_by(['ANVSAGEOID']).agg(MDF_Population = pl.len())
        anvsas_sex5yage =  hdfanvsas_sex5yage.join(mdfanvsas_sex5yage, on=['ANVSAGEOID'], how="full", coalesce=True).pipe(calculate_stats)
        ss = anvsas_sex5yage.pipe(calculate_ss, geography="ANVSA", sizecategory = "All", characteristic = "{sex} Age {agegroup}".format(sex = sexdict.get(s), agegroup = g))
        outputdflist.append(ss)

print("{} Sex By 5 Year Age Groups Done".format(datetime.now()))

# GQ 
for g in gqinstcats:
    hdfstates_gqinst = dfhdf().filter(pl.col('GQINST') == g).group_by(['TABBLKST']).agg(HDF_Population = pl.len())
    mdfstates_gqinst = dfmdf().filter(pl.col('GQINST') == g).group_by(['TABBLKST']).agg(MDF_Population = pl.len())
    states_gqinst =  hdfstates_gqinst.join(mdfstates_gqinst, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = states_gqinst.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)

for g in gqmajortypecats:
    hdfstates_gqmajtype = dfhdf().filter(pl.col('GQMAJORTYPE') == g).group_by(['TABBLKST']).agg(HDF_Population = pl.len())
    mdfstates_gqmajtype = dfmdf().filter(pl.col('GQMAJORTYPE') == g).group_by(['TABBLKST']).agg(MDF_Population = pl.len())
    states_gqmajtype =  hdfstates_gqmajtype.join(mdfstates_gqmajtype, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = states_gqmajtype.pipe(calculate_ss, geography="State", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)

for g in gqinstcats:
    hdfcounties_gqinst = dfhdf().filter(pl.col('GQINST') == g).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_gqinst = dfmdf().filter(pl.col('GQINST') == g).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_gqinst =  hdfcounties_gqinst.join(mdfcounties_gqinst, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = counties_gqinst.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)
    counties_gqinst = counties_gqinst.join(dfhdf().group_by('CountyGEOID').agg(pl.len().alias('HDF_TotalPopulation')).collect(), on="CountyGEOID", how="full", coalesce=True)
    counties_gqinst = counties_gqinst.with_columns(pl.col('HDF_TotalPopulation').cut([0,1000,5000,10000,50000,100000], left_closed=True).alias('Total_PopSize'))
    for i in counties_gqinst['Total_PopSize'].cat.get_categories():
        ss = counties_gqinst.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "GQ {}".format(g))
        outputdflist.append(ss)

for g in gqmajortypecats:
    hdfcounties_gqmajtype = dfhdf().filter(pl.col('GQMAJORTYPE') == g).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_gqmajtype = dfmdf().filter(pl.col('GQMAJORTYPE') == g).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_gqmajtype =  hdfcounties_gqmajtype.join(mdfcounties_gqmajtype, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = counties_gqmajtype.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)
    counties_gqmajtype = counties_gqmajtype.join(dfhdf().group_by('CountyGEOID').agg(pl.len().alias('HDF_TotalPopulation')).collect(), on="CountyGEOID", how="full", coalesce=True)
    counties_gqmajtype = counties_gqmajtype.with_columns(pl.col('HDF_TotalPopulation').cut([0,1000,5000,10000,50000,100000], left_closed=True).alias('Total_PopSize'))
    for i in counties_gqmajtype['Total_PopSize'].cat.get_categories():
        ss = counties_gqmajtype.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "GQ {}".format(g))
        outputdflist.append(ss)

for g in gqinstcats:
    hdfplaces_gqinst = dfhdf().filter(pl.col('GQINST') == g).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_gqinst = dfmdf().filter(pl.col('GQINST') == g).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_gqinst =  hdfplaces_gqinst.join(mdfplaces_gqinst, on=['IncPlaceGEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = places_gqinst.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)
    places_gqinst = places_gqinst.join(dfhdf().group_by('IncPlaceGEOID').agg(pl.len().alias('HDF_TotalPopulation')).join(pl.DataFrame({'IncPlaceGEOID': allplacesindex}), on='IncPlaceGEOID', how='right', coalesce=True).fill_null(0), on="IncPlaceGEOID", how="full", coalesce=True)
    places_gqinst = places_gqinst.with_columns(pl.col('HDF_TotalPopulation').cut([0,500,1000,5000,10000,50000,100000], left_closed=True).alias('Total_PopSize'))
    for i in places_gqinst['Total_PopSize'].cat.get_categories():
        ss = places_gqinst.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "GQ {}".format(g))
        outputdflist.append(ss)

for g in gqmajortypecats:
    hdfplaces_gqmajtype = dfhdf().filter(pl.col('GQMAJORTYPE') == g).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
    mdfplaces_gqmajtype = dfmdf().filter(pl.col('GQMAJORTYPE') == g).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
    places_gqmajtype =  hdfplaces_gqmajtype.join(mdfplaces_gqmajtype, on=['IncPlaceGEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = places_gqmajtype.pipe(calculate_ss, geography="Place", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)
    places_gqmajtype = places_gqmajtype.join(dfhdf().group_by('IncPlaceGEOID').agg(pl.len().alias('HDF_TotalPopulation')).join(pl.DataFrame({'IncPlaceGEOID': allplacesindex}), on='IncPlaceGEOID', how='right', coalesce=True).fill_null(0), on="IncPlaceGEOID", how="full", coalesce=True)
    places_gqmajtype = places_gqmajtype.with_columns(pl.col('HDF_TotalPopulation').cut([0,500,1000,5000,10000,50000,100000], left_closed=True).alias('Total_PopSize'))
    for i in places_gqmajtype['Total_PopSize'].cat.get_categories():
        ss = places_gqmajtype.filter(pl.col('Total_PopSize') == i).pipe(calculate_ss, geography="Place", sizecategory = str(i), characteristic = "GQ {}".format(g))
        outputdflist.append(ss)

for g in gqinstcats:
    hdftracts_gqinst = dfhdf().filter(pl.col('GQINST') == g).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
    mdftracts_gqinst = dfmdf().filter(pl.col('GQINST') == g).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
    tracts_gqinst =  hdftracts_gqinst.join(mdftracts_gqinst, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = tracts_gqinst.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)

for g in gqmajortypecats:
    hdftracts_gqmajtype = dfhdf().filter(pl.col('GQMAJORTYPE') == g).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
    mdftracts_gqmajtype = dfmdf().filter(pl.col('GQMAJORTYPE') == g).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
    tracts_gqmajtype =  hdftracts_gqmajtype.join(mdftracts_gqmajtype, on=['GEOID'], how="full", coalesce=True).pipe(calculate_stats)
    ss = tracts_gqmajtype.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "GQ {}".format(g))
    outputdflist.append(ss)

print("{} GQ Types Done".format(datetime.now()))



# Counties Absolute Change in Median Age/Sex Ratio 
hdfcounties_medage = dfhdf().group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_MedianAge')).collect().join(pl.DataFrame({'GEOID': allcountiesindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
mdfcounties_medage = dfmdf().group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_MedianAge')).collect().join(pl.DataFrame({'GEOID': allcountiesindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
counties_medage =  hdfcounties_medage.join(mdfcounties_medage, on='CountyGEOID', how='full', coalesce=True)
counties_medage = counties_medage.with_columns((pl.col('HDF_MedianAge') - pl.col('MDF_MedianAge')).abs().alias('AbsDiffMedAge'))
counties_medage.write_csv(f"{OUTPUTDIR}/counties_medage.csv")
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'Average Absolute Change in Median Age','NumCells':len(counties_medage),'AvgAbsDiffMedAge': counties_medage['AbsDiffMedAge'].drop_nulls().mean()})
outputdflist.append(ss)
counties_medage = counties_medage.join(dfhdf().group_by('CountyGEOID').agg(pl.len().alias('HDF_TotalPopulation')).collect(), on="CountyGEOID", how="full", coalesce=True)
counties_medage = counties_medage.with_columns(pl.col('HDF_TotalPopulation').cut([0,1000,5000,10000,50000,100000], left_closed=True).alias('Total_PopSize'))
for i in counties_medage['Total_PopSize'].cat.get_categories():
    ss = pl.DataFrame({'Geography':'County', 'Size_Category':str(i), 'Characteristic':'Average Absolute Change in Median Age','NumCells':len(counties_medage.filter(pl.col('Total_PopSize') == i)),'AvgAbsDiffMedAge': counties_medage.filter(pl.col('Total_PopSize') == i).get_column('AbsDiffMedAge').drop_nulls().mean()})
    outputdflist.append(ss)

hdfcounties_sexratio = dfhdf().group_by(['CountyGEOID', 'QSEX']).agg(pl.len().alias('count')).collect().pivot(on='QSEX', index='CountyGEOID', values='count').fill_null(0).join(pl.DataFrame({'CountyGEOID': allcountiesindex}), on='CountyGEOID', how='right', coalesce=True).fill_null(0).with_columns((100 * pl.col('1') / pl.col('2')).alias('HDF_SexRatio')).drop(['1', '2'])
mdfcounties_sexratio = dfmdf().group_by(['CountyGEOID', 'QSEX']).agg(pl.len().alias('count')).collect().pivot(on='QSEX', index='CountyGEOID', values='count').fill_null(0).join(pl.DataFrame({'CountyGEOID': allcountiesindex}), on='CountyGEOID', how='right', coalesce=True).fill_null(0).with_columns((100 * pl.col('1') / pl.col('2')).alias('MDF_SexRatio')).drop(['1', '2'])
counties_sexratio =  hdfcounties_sexratio.join(mdfcounties_sexratio, on='CountyGEOID', how='full', coalesce=True)
counties_sexratio = counties_sexratio.with_columns((pl.col('HDF_SexRatio') - pl.col('MDF_SexRatio')).abs().alias('AbsDiffSexRatio'))
counties_sexratio.write_csv(f"{OUTPUTDIR}/counties_sexratio.csv")
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'Average Absolute Change in Sex Ratio','NumCells':len(counties_sexratio),'AvgAbsDiffSexRatio': counties_sexratio['AbsDiffSexRatio'].drop_nulls().mean()})
outputdflist.append(ss)
counties_sexratio = counties_sexratio.join(dfhdf().group_by('CountyGEOID').agg(pl.len().alias('HDF_TotalPopulation')).collect(), on="CountyGEOID", how="full", coalesce=True)
counties_sexratio = counties_sexratio.with_columns(pl.col('HDF_TotalPopulation').cut([0,1000,5000,10000,50000,100000], left_closed=True).alias('Total_PopSize'))
for i in counties_sexratio['Total_PopSize'].cat.get_categories():
    ss = pl.DataFrame({'Geography':'County', 'Size_Category':str(i), 'Characteristic':'Average Absolute Change in Sex Ratio','NumCells':len(counties_sexratio.filter(pl.col('Total_PopSize') == i)),'AvgAbsDiffSexRatio': counties_sexratio.filter(pl.col('Total_PopSize') == i).get_column('AbsDiffSexRatio').drop_nulls().mean()})
    outputdflist.append(ss)


# Counties GQ Absolute Change in Median Age/Sex Ratio 
hdfcountiesgq_medage = dfhdf().filter(pl.col('GQTYPE') > 0).group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_MedianAge')).collect().join(pl.DataFrame({'GEOID': allcountiesindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
mdfcountiesgq_medage = dfmdf().filter(pl.col('GQTYPE') > 0).group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_MedianAge')).collect().join(pl.DataFrame({'GEOID': allcountiesindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
countiesgq_medage =  hdfcountiesgq_medage.join(mdfcountiesgq_medage, on='CountyGEOID', how='full', coalesce=True)
countiesgq_medage = countiesgq_medage.with_columns((pl.col('HDF_MedianAge') - pl.col('MDF_MedianAge')).abs().alias('AbsDiffMedAge'))
countiesgq_medage.write_csv(f"{OUTPUTDIR}/countiesgq_medage.csv")
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'Average Absolute Change in Median Age of GQ Population','NumCells':len(countiesgq_medage),'AvgAbsDiffMedAge': countiesgq_medage['AbsDiffMedAge'].drop_nulls().mean()})
outputdflist.append(ss)
countiesgq_medage = countiesgq_medage.join(dfhdf().group_by('CountyGEOID').agg(pl.len().alias('HDF_TotalPopulation')).collect(), on="CountyGEOID", how="full", coalesce=True)
countiesgq_medage = countiesgq_medage.with_columns(pl.col('HDF_TotalPopulation').cut([0,1000,5000,10000,50000,100000], left_closed=True).alias('Total_PopSize'))
for i in countiesgq_medage['Total_PopSize'].cat.get_categories():
    ss = pl.DataFrame({'Geography':'County', 'Size_Category':str(i), 'Characteristic':'Average Absolute Change in Median Age of GQ Population','NumCells':len(countiesgq_medage.filter(pl.col('Total_PopSize') == i)),'AvgAbsDiffMedAge': countiesgq_medage.filter(pl.col('Total_PopSize') == i).get_column('AbsDiffMedAge').drop_nulls().mean()})
    outputdflist.append(ss)

hdfcountiesgq_sexratio = dfhdf().filter(pl.col('GQTYPE') > 0).group_by(['CountyGEOID', 'QSEX']).agg(pl.len().alias('count')).collect().pivot(on='QSEX', index='CountyGEOID', values='count').fill_null(0).join(pl.DataFrame({'CountyGEOID': allcountiesindex}), on='CountyGEOID', how='right', coalesce=True).fill_null(0).with_columns((100 * pl.col('1') / pl.col('2')).alias('HDF_SexRatio')).drop(['1', '2'])
mdfcountiesgq_sexratio = dfmdf().filter(pl.col('GQTYPE') > 0).group_by(['CountyGEOID', 'QSEX']).agg(pl.len().alias('count')).collect().pivot(on='QSEX', index='CountyGEOID', values='count').fill_null(0).join(pl.DataFrame({'CountyGEOID': allcountiesindex}), on='CountyGEOID', how='right', coalesce=True).fill_null(0).with_columns((100 * pl.col('1') / pl.col('2')).alias('MDF_SexRatio')).drop(['1', '2'])
countiesgq_sexratio =  hdfcountiesgq_sexratio.join(mdfcountiesgq_sexratio, on='CountyGEOID', how='full', coalesce=True)
countiesgq_sexratio = countiesgq_sexratio.with_columns((pl.col('HDF_SexRatio') - pl.col('MDF_SexRatio')).abs().alias('AbsDiffSexRatio'))
countiesgq_sexratio.write_csv(f"{OUTPUTDIR}/countiesgq_sexratio.csv")
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'Average Absolute Change in Sex Ratio of GQ Population','NumCells':len(countiesgq_sexratio),'AvgAbsDiffSexRatio': countiesgq_sexratio['AbsDiffSexRatio'].drop_nulls().mean()})
outputdflist.append(ss)
countiesgq_sexratio = countiesgq_sexratio.join(dfhdf().group_by('CountyGEOID').agg(pl.len().alias('HDF_TotalPopulation')).collect(), on="CountyGEOID", how="full", coalesce=True)
countiesgq_sexratio = countiesgq_sexratio.with_columns(pl.col('HDF_TotalPopulation').cut([0,1000,5000,10000,50000,100000], left_closed=True).alias('Total_PopSize'))
for i in countiesgq_sexratio['Total_PopSize'].cat.get_categories():
    ss = pl.DataFrame({'Geography':'County', 'Size_Category':str(i), 'Characteristic':'Average Absolute Change in Sex Ratio of GQ Population','NumCells':len(countiesgq_sexratio.filter(pl.col('Total_PopSize') == i)),'AvgAbsDiffSexRatio': countiesgq_sexratio.filter(pl.col('Total_PopSize') == i).get_column('AbsDiffSexRatio').drop_nulls().mean()})
    outputdflist.append(ss)

print("{} Average Absolute Change in Median Age and Sex Ratio Done".format(datetime.now()))


print("{} Starting Use Cases".format(datetime.now()))

# Tracts Aged 75+
hdftracts_over75 = dfhdf().filter(pl.col('QAGE') >= 75).group_by('TractGEOID').agg(HDF_Population = pl.len())
mdftracts_over75 = dfmdf().filter(pl.col('QAGE') >= 75).group_by('TractGEOID').agg(MDF_Population = pl.len())
tracts_over75 =  hdftracts_over75.join(mdftracts_over75, on='GEOID', how="full", coalesce=True).pipe(calculate_stats)
ss = tracts_over75.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "Aged 75 and Over")
outputdflist.append(ss)

# Counties and Places By State TAES
for s in allstates:
    dfhdfstate = dfhdf().filter(pl.col('TABBLKST') == s)
    dfmdfstate = dfmdf().filter(pl.col('TABBLKST') == s)
    hdfcounties_taes = dfhdfstate.group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_taes = dfmdfstate.group_by('CountyGEOID').agg(MDF_Population = pl.len())
    hdfsize = len(dfhdfstate)
    mdfsize = len(dfmdfstate)
    counties_taes =  hdfcounties_taes.join(mdfcounties_taes, on='GEOID', how="full", coalesce=True)
    counties_taes = counties_taes.with_columns(pl.col('HDF_Population').fill_null(0), pl.col('MDF_Population').fill_null(0))
    counties_taes  = counties_taes.with_columns(((pl.col('HDF_Population')/hdfsize - pl.col('MDF_Population')/mdfsize).abs()).alias('AES'))
    ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'TAES Total Population {}'.format(statedict.get(s)),'NumCells':len(counties_taes),'TAES': counties_taes['AES'].sum()})
    outputdflist.append(ss)
    if s == "15":
        ss = pl.DataFrame({'Geography':'Place', 'Size_Category':'All', 'Characteristic':'TAES Total Population {}'.format(statedict.get(s)),'NumCells':0,'TAES': 0})
        outputdflist.append(ss)
    else:
        hdfplaces_taes = dfhdfstate.group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
        mdfplaces_taes = dfmdfstate.group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
        places_taes =  hdfplaces_taes.join(mdfplaces_taes, on='IncPlaceGEOID', how="full", coalesce=True)
        places_taes = places_taes.with_columns(pl.col('HDF_Population').fill_null(0), pl.col('MDF_Population').fill_null(0))
        places_taes  = places_taes.with_columns(((pl.col('HDF_Population')/hdfsize - pl.col('MDF_Population')/mdfsize).abs()).alias('AES'))
        ss = pl.DataFrame({'Geography':'Place', 'Size_Category':'All', 'Characteristic':'TAES Total Population {}'.format(statedict.get(s)),'NumCells':len(places_taes),'TAES': places_taes['AES'].sum()})
        outputdflist.append(ss)

# MCDs By State TAES
for s in mcdstates:
    dfhdfstate = dfhdf().filter(pl.col('TABBLKST') == s)
    dfmdfstate = dfmdf().filter(pl.col('TABBLKST') == s)
    hdfmcds_taes = dfhdfstate.group_by(['MCDGEOID']).agg(HDF_Population = pl.len())
    mdfmcds_taes = dfmdfstate.group_by(['MCDGEOID']).agg(MDF_Population = pl.len())
    hdfsize = len(dfhdfstate)
    mdfsize = len(dfmdfstate)
    mcds_taes =  hdfmcds_taes.join(mdfmcds_taes, on='MCDGEOID', how="full", coalesce=True)
    mcds_taes = mcds_taes.with_columns(pl.col('HDF_Population').fill_null(0), pl.col('MDF_Population').fill_null(0))
    mcds_taes  = mcds_taes.with_columns(((pl.col('HDF_Population')/hdfsize - pl.col('MDF_Population')/mdfsize).abs()).alias('AES'))
    ss = pl.DataFrame({'Geography':'MCD', 'Size_Category':'All', 'Characteristic':'TAES Total Population {}'.format(statedict.get(s)),'NumCells':len(mcds_taes),'TAES': mcds_taes['AES'].sum()})
    outputdflist.append(ss)

# Counties Single Year of Age < 18  
for y in list(range(0,18)):
    hdfcounties_age = dfhdf().filter(pl.col('QAGE') == y).group_by('CountyGEOID').agg(HDF_Population = pl.len())
    mdfcounties_age = dfmdf().filter(pl.col('QAGE') == y).group_by('CountyGEOID').agg(MDF_Population = pl.len())
    counties_age =  hdfcounties_age.join(mdfcounties_age, on='GEOID', how="full", coalesce=True).pipe(calculate_stats)
    ss = counties_age.pipe(calculate_ss, geography="County", sizecategory = "All", characteristic = "Age {}".format(y))
    outputdflist.append(ss)
    counties_age = counties_age.join(dfhdf().filter(pl.col('QAGE') < 18).group_by('CountyGEOID').agg(pl.len().alias('HDF_PopulationUnder18')), on="GEOID", how="full", coalesce=True)
    counties_age = counties_age.with_columns(pl.col('HDF_PopulationUnder18').cut([0,1000,10000], left_closed=True).alias('Under18_PopSize'))
    for i in counties_age['Under18_PopSize'].cat.get_categories():
        ss = counties_age.filter(pl.col('Under18_PopSize') == i).pipe(calculate_ss, geography="County", sizecategory = str(i), characteristic = "Age {}".format(y))
        outputdflist.append(ss)

# Elem School Districts Single Year of Age < 18  
for y in list(range(0,18)):
    hdfelemschdists_age = dfhdf().filter(pl.col('QAGE') == y).group_by(['SchDistEGEOID']).agg(HDF_Population = pl.len())
    mdfelemschdists_age = dfmdf().filter(pl.col('QAGE') == y).group_by(['SchDistEGEOID']).agg(MDF_Population = pl.len())
    elemschdists_age =  hdfelemschdists_age.join(mdfelemschdists_age, on='SchDistEGEOID', how="full", coalesce=True).pipe(calculate_stats)
    ss = elemschdists_age.pipe(calculate_ss, geography="ESD", sizecategory = "All", characteristic = "Age {}".format(y))
    outputdflist.append(ss)
    elemschdists_age = elemschdists_age.join(dfhdf().filter(pl.col('QAGE') < 18).group_by('SchDistEGEOID').agg(pl.len().alias('HDF_PopulationUnder18')).collect().join(pl.DataFrame({'SchDistEGEOID': allelemschdistsindex}), on='SchDistEGEOID', how='right', coalesce=True).fill_null(0), on="SchDistEGEOID", how="full", coalesce=True)
    elemschdists_age = elemschdists_age.with_columns(pl.col('HDF_PopulationUnder18').cut([0,1000,10000], left_closed=True).alias('Under18_PopSize'))
    for i in elemschdists_age['Under18_PopSize'].cat.get_categories():
        ss = elemschdists_age.filter(pl.col('Under18_PopSize') == i).pipe(calculate_ss, geography="ESD", sizecategory = str(i), characteristic = "Age {}".format(y))
        outputdflist.append(ss)

# Sec School Districts Single Year of Age < 18  
for y in list(range(0,18)):
    hdfsecschdists_age = dfhdf().filter(pl.col('QAGE') == y).group_by(['SchDistSGEOID']).agg(HDF_Population = pl.len())
    mdfsecschdists_age = dfmdf().filter(pl.col('QAGE') == y).group_by(['SchDistSGEOID']).agg(MDF_Population = pl.len())
    secschdists_age =  hdfsecschdists_age.join(mdfsecschdists_age, on='SchDistSGEOID', how="full", coalesce=True).pipe(calculate_stats)
    ss = secschdists_age.pipe(calculate_ss, geography="SSD", sizecategory = "All", characteristic = "Age {}".format(y))
    outputdflist.append(ss)
    secschdists_age = secschdists_age.join(dfhdf().filter(pl.col('QAGE') < 18).group_by('SchDistSGEOID').agg(pl.len().alias('HDF_PopulationUnder18')).collect().join(pl.DataFrame({'SchDistSGEOID': allsecschdistsindex}), on='SchDistSGEOID', how='right', coalesce=True).fill_null(0), on="SchDistSGEOID", how="full", coalesce=True)
    secschdists_age = secschdists_age.with_columns(pl.col('HDF_PopulationUnder18').cut([0,1000,10000], left_closed=True).alias('Under18_PopSize'))
    for i in secschdists_age['Under18_PopSize'].cat.get_categories():
        ss = secschdists_age.filter(pl.col('Under18_PopSize') == i).pipe(calculate_ss, geography="SSD", sizecategory = str(i), characteristic = "Age {}".format(y))
        outputdflist.append(ss)

# Uni School Districts Single Year of Age < 18  
for y in list(range(0,18)):
    hdfunischdists_age = dfhdf().filter(pl.col('QAGE') == y).group_by(['SchDistUGEOID']).agg(HDF_Population = pl.len())
    mdfunischdists_age = dfmdf().filter(pl.col('QAGE') == y).group_by(['SchDistUGEOID']).agg(MDF_Population = pl.len())
    unischdists_age =  hdfunischdists_age.join(mdfunischdists_age, on='SchDistUGEOID', how="full", coalesce=True).pipe(calculate_stats)
    ss = unischdists_age.pipe(calculate_ss, geography="USD", sizecategory = "All", characteristic = "Age {}".format(y))
    outputdflist.append(ss)
    unischdists_age = unischdists_age.join(dfhdf().filter(pl.col('QAGE') < 18).group_by('SchDistUGEOID').agg(pl.len().alias('HDF_PopulationUnder18')).collect().join(pl.DataFrame({'SchDistUGEOID': allunischdistsindex}), on='SchDistUGEOID', how='right', coalesce=True).fill_null(0), on="SchDistUGEOID", how="full", coalesce=True)
    unischdists_age = unischdists_age.with_columns(pl.col('HDF_PopulationUnder18').cut([0,1000,10000], left_closed=True).alias('Under18_PopSize'))
    for i in unischdists_age['Under18_PopSize'].cat.get_categories():
        ss = unischdists_age.filter(pl.col('Under18_PopSize') == i).pipe(calculate_ss, geography="USD", sizecategory = str(i), characteristic = "Age {}".format(y))
        outputdflist.append(ss)

# Counties Nationwide AIAN Alone or In Combination TAES
hdfcounties_aianaloneorincomb_taes = dfhdf[dfhdf['aianalone-or-incomb'] == 1].group_by('CountyGEOID').agg(HDF_Population = pl.len())
mdfcounties_aianaloneorincomb_taes = dfmdf[dfmdf['aianalone-or-incomb'] == 1].group_by('CountyGEOID').agg(MDF_Population = pl.len())
hdfsize_aianaloneorincomb = len(dfhdf[dfhdf['aianalone-or-incomb'] == 1])
mdfsize_aianaloneorincomb = len(dfmdf[dfmdf['aianalone-or-incomb'] == 1])
counties_aianaloneorincomb_taes =  hdfcounties_aianaloneorincomb_taes.join(mdfcounties_aianaloneorincomb_taes, on='GEOID', how="full", coalesce=True)
counties_aianaloneorincomb_taes  = counties_aianaloneorincomb_taes.with_columns(((pl.col('HDF_Population')/hdfsize_aianaloneorincomb - pl.col('MDF_Population')/mdfsize_aianaloneorincomb).abs()).alias('AES'))
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'TAES AIAN Alone or In Combination Nation','NumCells':len(counties_aianaloneorincomb_taes),'TAES': counties_aianaloneorincomb_taes['AES'].sum()})
outputdflist.append(ss)

# Places Nationwide AIAN Alone or In Combination TAES
hdfplaces_aianaloneorincomb_taes = dfhdf[dfhdf['aianalone-or-incomb'] == 1].group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
mdfplaces_aianaloneorincomb_taes = dfmdf[dfmdf['aianalone-or-incomb'] == 1].group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
# hdfsize_aianaloneorincomb = len(dfhdf[dfhdf['aianalone-or-incomb'] == 1])
# mdfsize_aianaloneorincomb = len(dfmdf[dfmdf['aianalone-or-incomb'] == 1])
places_aianaloneorincomb_taes =  hdfplaces_aianaloneorincomb_taes.join(mdfplaces_aianaloneorincomb_taes, on='IncPlaceGEOID', how="full", coalesce=True)
places_aianaloneorincomb_taes  = places_aianaloneorincomb_taes.with_columns(((pl.col('HDF_Population')/hdfsize_aianaloneorincomb - pl.col('MDF_Population')/mdfsize_aianaloneorincomb).abs()).alias('AES'))
ss = pl.DataFrame({'Geography':'Place', 'Size_Category':'All', 'Characteristic':'TAES AIAN Alone or In Combination Nation','NumCells':len(places_aianaloneorincomb_taes),'TAES': places_aianaloneorincomb_taes['AES'].sum()})
outputdflist.append(ss)

# Counties AIAN Alone Count Where MDF < HDF
hdfcounties_aianalone = dfhdf().filter(pl.col('CENRACE') == 3).group_by('CountyGEOID').agg(HDF_Population = pl.len())
mdfcounties_aianalone = dfmdf().filter(pl.col('CENRACE') == 3).group_by('CountyGEOID').agg(MDF_Population = pl.len())
counties_aianalone =  hdfcounties_aianalone.join(mdfcounties_aianalone, on='GEOID', how="full", coalesce=True).pipe(calculate_stats)
counties_aianalone = counties_aianalone.with_columns(pl.when(pl.col('MDF_Population')  < pl.col('HDF_Population')).then(1).otherwise(0).alias('MDFltHDF'))
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(counties_aianalone),'CountMDFltHDF': counties_aianalone['MDFltHDF'].sum(), 'MedianPctDiffWhereMDFltHDF':counties_aianalone.filter(pl.col('MDFltHDF') == 1).get_column('PercDiff').drop_nulls().median()})
outputdflist.append(ss)
counties_aianalone = counties_aianalone.with_columns(pl.col('HDF_Population').cut([0,10,100], left_closed=True).alias('AIAN_PopSize'))
for i in counties_aianalone['AIAN_PopSize'].cat.get_categories():
    ss = pl.DataFrame({'Geography':'County', 'Size_Category':'AIAN Population Size {}'.format(i), 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(counties_aianalone.filter(pl.col('AIAN_PopSize') == i)),'CountMDFltHDF': counties_aianalone.filter(pl.col('AIAN_PopSize') == i).get_column('MDFltHDF').sum(), 'MedianPctDiffWhereMDFltHDF':counties_aianalone.filter((pl.col('AIAN_PopSize') == i) & (pl.col('MDFltHDF') == 1)).get_column('PercDiff').drop_nulls().median()})
    outputdflist.append(ss)


# Places AIAN Alone Count Where MDF < HDF
hdfplaces_aianalone = dfhdf().filter(pl.col('CENRACE') == 3).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
mdfplaces_aianalone = dfmdf().filter(pl.col('CENRACE') == 3).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
places_aianalone =  hdfplaces_aianalone.join(mdfplaces_aianalone, on='IncPlaceGEOID', how="full", coalesce=True).pipe(calculate_stats)
places_aianalone = places_aianalone.with_columns(pl.when(pl.col('MDF_Population')  < pl.col('HDF_Population')).then(1).otherwise(0).alias('MDFltHDF'))
ss = pl.DataFrame({'Geography':'Place', 'Size_Category':'All', 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(places_aianalone),'CountMDFltHDF': places_aianalone['MDFltHDF'].sum(), 'MedianPctDiffWhereMDFltHDF':places_aianalone.filter(pl.col('MDFltHDF') == 1).get_column('PercDiff').drop_nulls().median()})
outputdflist.append(ss)
places_aianalone = places_aianalone.with_columns(pl.col('HDF_Population').cut([0,10,100], left_closed=True).alias('AIAN_PopSize'))
for i in places_aianalone['AIAN_PopSize'].cat.get_categories():
    ss = pl.DataFrame({'Geography':'Place', 'Size_Category':'AIAN Population Size {}'.format(i), 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(places_aianalone.filter(pl.col('AIAN_PopSize') == i)),'CountMDFltHDF': places_aianalone.filter(pl.col('AIAN_PopSize') == i).get_column('MDFltHDF').sum(), 'MedianPctDiffWhereMDFltHDF':places_aianalone.filter((pl.col('AIAN_PopSize') == i) & (pl.col('MDFltHDF') == 1)).get_column('PercDiff').drop_nulls().median()})
    outputdflist.append(ss)


# Fed AIR AIAN Alone Count Where MDF < HDF
hdffedairs_aianalone = dfhdf().filter(pl.col('CENRACE') == 3).group_by(['FedAIRGEOID']).agg(HDF_Population = pl.len())
mdffedairs_aianalone = dfmdf().filter(pl.col('CENRACE') == 3).group_by(['FedAIRGEOID']).agg(MDF_Population = pl.len())
fedairs_aianalone =  hdffedairs_aianalone.join(mdffedairs_aianalone, on='FedAIRGEOID', how="full", coalesce=True).pipe(calculate_stats)
fedairs_aianalone = fedairs_aianalone.with_columns(pl.when(pl.col('MDF_Population')  < pl.col('HDF_Population')).then(1).otherwise(0).alias('MDFltHDF'))
ss = pl.DataFrame({'Geography':'Fed AIR', 'Size_Category':'All', 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(fedairs_aianalone),'CountMDFltHDF': fedairs_aianalone['MDFltHDF'].sum(), 'MedianPctDiffWhereMDFltHDF':fedairs_aianalone.filter(pl.col('MDFltHDF') == 1).get_column('PercDiff').drop_nulls().median()})
outputdflist.append(ss)
fedairs_aianalone = fedairs_aianalone.with_columns(pl.col('HDF_Population').cut([0,10,100], left_closed=True).alias('AIAN_PopSize'))
for i in fedairs_aianalone['AIAN_PopSize'].cat.get_categories():
    ss = pl.DataFrame({'Geography':'Fed AIR', 'Size_Category':'AIAN Population Size {}'.format(i), 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(fedairs_aianalone.filter(pl.col('AIAN_PopSize') == i)),'CountMDFltHDF': fedairs_aianalone.filter(pl.col('AIAN_PopSize') == i).get_column('MDFltHDF').sum(), 'MedianPctDiffWhereMDFltHDF':fedairs_aianalone.filter((pl.col('AIAN_PopSize') == i) & (pl.col('MDFltHDF') == 1)).get_column('PercDiff').drop_nulls().median()})
    outputdflist.append(ss)


# OTSA AIAN Alone Count Where MDF < HDF
hdfotsas_aianalone = dfhdf().filter(pl.col('CENRACE') == 3).group_by(['OTSAGEOID']).agg(HDF_Population = pl.len())
mdfotsas_aianalone = dfmdf().filter(pl.col('CENRACE') == 3).group_by(['OTSAGEOID']).agg(MDF_Population = pl.len())
otsas_aianalone =  hdfotsas_aianalone.join(mdfotsas_aianalone, on='OTSAGEOID', how="full", coalesce=True).pipe(calculate_stats)
otsas_aianalone = otsas_aianalone.with_columns(pl.when(pl.col('MDF_Population')  < pl.col('HDF_Population')).then(1).otherwise(0).alias('MDFltHDF'))
ss = pl.DataFrame({'Geography':'OTSA', 'Size_Category':'All', 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(otsas_aianalone),'CountMDFltHDF': otsas_aianalone['MDFltHDF'].sum(), 'MedianPctDiffWhereMDFltHDF':otsas_aianalone.filter(pl.col('MDFltHDF') == 1).get_column('PercDiff').drop_nulls().median()})
outputdflist.append(ss)
otsas_aianalone = otsas_aianalone.with_columns(pl.col('HDF_Population').cut([0,10,100], left_closed=True).alias('AIAN_PopSize'))
for i in otsas_aianalone['AIAN_PopSize'].cat.get_categories():
    ss = pl.DataFrame({'Geography':'OTSA', 'Size_Category':'AIAN Population Size {}'.format(i), 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(otsas_aianalone.filter(pl.col('AIAN_PopSize') == i)),'CountMDFltHDF': otsas_aianalone.filter(pl.col('AIAN_PopSize') == i).get_column('MDFltHDF').sum(), 'MedianPctDiffWhereMDFltHDF':otsas_aianalone.filter((pl.col('AIAN_PopSize') == i) & (pl.col('MDFltHDF') == 1)).get_column('PercDiff').drop_nulls().median()})
    outputdflist.append(ss)

# ANVSA AIAN Alone Count Where MDF < HDF
hdfanvsas_aianalone = dfhdf().filter(pl.col('CENRACE') == 3).group_by(['ANVSAGEOID']).agg(HDF_Population = pl.len())
mdfanvsas_aianalone = dfmdf().filter(pl.col('CENRACE') == 3).group_by(['ANVSAGEOID']).agg(MDF_Population = pl.len())
anvsas_aianalone =  hdfanvsas_aianalone.join(mdfanvsas_aianalone, on='ANVSAGEOID', how="full", coalesce=True).pipe(calculate_stats)
anvsas_aianalone = anvsas_aianalone.with_columns(pl.when(pl.col('MDF_Population')  < pl.col('HDF_Population')).then(1).otherwise(0).alias('MDFltHDF'))
ss = pl.DataFrame({'Geography':'ANVSA', 'Size_Category':'All', 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(anvsas_aianalone),'CountMDFltHDF': anvsas_aianalone['MDFltHDF'].sum(), 'MedianPctDiffWhereMDFltHDF':anvsas_aianalone.filter(pl.col('MDFltHDF') == 1).get_column('PercDiff').drop_nulls().median()})
outputdflist.append(ss)
anvsas_aianalone = anvsas_aianalone.with_columns(pl.col('HDF_Population').cut([0,10,100], left_closed=True).alias('AIAN_PopSize'))
for i in anvsas_aianalone['AIAN_PopSize'].cat.get_categories():
    ss = pl.DataFrame({'Geography':'ANVSA', 'Size_Category':'AIAN Population Size {}'.format(i), 'Characteristic':'AIAN Alone CountMDFltHDF', 'NumCells':len(anvsas_aianalone.filter(pl.col('AIAN_PopSize') == i)),'CountMDFltHDF': anvsas_aianalone.filter(pl.col('AIAN_PopSize') == i).get_column('MDFltHDF').sum(), 'MedianPctDiffWhereMDFltHDF':anvsas_aianalone.filter((pl.col('AIAN_PopSize') == i) & (pl.col('MDFltHDF') == 1)).get_column('PercDiff').drop_nulls().median()})
    outputdflist.append(ss)

# Counties NHPI Alone Count Where MDF < HDF
hdfcounties_nhpialone = dfhdf().filter(pl.col('CENRACE') == 5).group_by('CountyGEOID').agg(HDF_Population = pl.len())
mdfcounties_nhpialone = dfmdf().filter(pl.col('CENRACE') == 5).group_by('CountyGEOID').agg(MDF_Population = pl.len())
counties_nhpialone =  hdfcounties_nhpialone.join(mdfcounties_nhpialone, on='GEOID', how="full", coalesce=True).pipe(calculate_stats)
counties_nhpialone = counties_nhpialone.with_columns(pl.when(pl.col('MDF_Population')  < pl.col('HDF_Population')).then(1).otherwise(0).alias('MDFltHDF'))
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'NHPI Alone CountMDFltHDF', 'NumCells':len(counties_nhpialone),'CountMDFltHDF': counties_nhpialone['MDFltHDF'].sum(), 'MedianPctDiffWhereMDFltHDF':counties_nhpialone.filter(pl.col('MDFltHDF') == 1).get_column('PercDiff').drop_nulls().median()})
outputdflist.append(ss)
counties_nhpialone = counties_nhpialone.with_columns(pl.col('HDF_Population').cut([0,10,100], left_closed=True).alias('NHPI_PopSize'))
for i in counties_nhpialone['NHPI_PopSize'].cat.get_categories():
    ss = pl.DataFrame({'Geography':'County', 'Size_Category':'NHPI Population Size {}'.format(i), 'Characteristic':'NHPI Alone CountMDFltHDF', 'NumCells':len(counties_nhpialone.filter(pl.col('NHPI_PopSize') == i)),'CountMDFltHDF': counties_nhpialone.filter(pl.col('NHPI_PopSize') == i).get_column('MDFltHDF').sum(), 'MedianPctDiffWhereMDFltHDF':counties_nhpialone.filter((pl.col('NHPI_PopSize') == i) & (pl.col('MDFltHDF') == 1)).get_column('PercDiff').drop_nulls().median()})
    outputdflist.append(ss)

# Places NHPI Alone Count Where MDF < HDF
hdfplaces_nhpialone = dfhdf().filter(pl.col('CENRACE') == 5).group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
mdfplaces_nhpialone = dfmdf().filter(pl.col('CENRACE') == 5).group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
places_nhpialone =  hdfplaces_nhpialone.join(mdfplaces_nhpialone, on='IncPlaceGEOID', how="full", coalesce=True).pipe(calculate_stats)
places_nhpialone = places_nhpialone.with_columns(pl.when(pl.col('MDF_Population')  < pl.col('HDF_Population')).then(1).otherwise(0).alias('MDFltHDF'))
ss = pl.DataFrame({'Geography':'Place', 'Size_Category':'All', 'Characteristic':'NHPI Alone CountMDFltHDF', 'NumCells':len(places_nhpialone),'CountMDFltHDF': places_nhpialone['MDFltHDF'].sum(), 'MedianPctDiffWhereMDFltHDF':places_nhpialone.filter(pl.col('MDFltHDF') == 1).get_column('PercDiff').drop_nulls().median()})
outputdflist.append(ss)
places_nhpialone = places_nhpialone.with_columns(pl.col('HDF_Population').cut([0,10,100], left_closed=True).alias('NHPI_PopSize'))
for i in places_nhpialone['NHPI_PopSize'].cat.get_categories():
    ss = pl.DataFrame({'Geography':'Place', 'Size_Category':'NHPI Population Size {}'.format(i), 'Characteristic':'NHPI Alone CountMDFltHDF', 'NumCells':len(places_nhpialone.filter(pl.col('NHPI_PopSize') == i)),'CountMDFltHDF': places_nhpialone.filter(pl.col('NHPI_PopSize') == i).get_column('MDFltHDF').sum(), 'MedianPctDiffWhereMDFltHDF':places_nhpialone.filter((pl.col('NHPI_PopSize') == i) & (pl.col('MDFltHDF') == 1)).get_column('PercDiff').drop_nulls().median()})
    outputdflist.append(ss)

# Tracts AIAN Alone or in Combination 
hdftracts_aianincomb = dfhdf().filter((dfhdf['aianalone-or-incomb']==1).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
mdftracts_aianincomb = dfmdf().filter((dfmdf['aianalone-or-incomb']==1).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
tracts_aianincomb =  hdftracts_aianincomb.join(mdftracts_aianincomb, on='GEOID', how="full", coalesce=True).pipe(calculate_stats)
tracts_aianincomb = tracts_aianincomb.with_columns(pl.when((pl.col('HDF_Population') < 20) & (pl.col('MDF_Population') >=100)).then(1).otherwise(0).alias('HundredPlusMDFLessThan20HDF'))
tracts_aianincomb = tracts_aianincomb.with_columns(pl.when((pl.col('HDF_Population') >=100) & (pl.col('MDF_Population') < 20)).then(1).otherwise(0).alias('LessThan20MDFHundredPlusHDF'))
ss = tracts_aianincomb.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "AIAN Alone Or In Combination")
ss = ss.hstack(pl.DataFrame({'Number100PlusMDFLessThan20HDF':tracts_aianincomb['HundredPlusMDFLessThan20HDF'].sum(), 'NumberLessThan20MDF100PlusHDF':tracts_aianincomb['LessThan20MDFHundredPlusHDF'].sum()}))
outputdflist.append(ss)

# Tracts NHPI
hdftracts_nhpialone = dfhdf().filter(pl.col('CENRACE') == 5).group_by(['TractGEOID']).agg(HDF_Population = pl.len())
mdftracts_nhpialone = dfmdf().filter(pl.col('CENRACE') == 5).group_by(['TractGEOID']).agg(MDF_Population = pl.len())
tracts_nhpialone =  hdftracts_nhpialone.join(mdftracts_nhpialone, on='GEOID', how="full", coalesce=True).pipe(calculate_stats)
tracts_nhpialone = tracts_nhpialone.with_columns(pl.when((pl.col('HDF_Population') < 20) & (pl.col('MDF_Population') >=100)).then(1).otherwise(0).alias('HundredPlusMDFLessThan20HDF'))
tracts_nhpialone = tracts_nhpialone.with_columns(pl.when((pl.col('HDF_Population') >=100) & (pl.col('MDF_Population') < 20)).then(1).otherwise(0).alias('LessThan20MDFHundredPlusHDF'))
ss = tracts_nhpialone.pipe(calculate_ss, geography="Tract", sizecategory = "All", characteristic = "NHPI Alone")
ss = ss.hstack(pl.DataFrame({'Number100PlusMDFLessThan20HDF':tracts_nhpialone['HundredPlusMDFLessThan20HDF'].sum(), 'NumberLessThan20MDF100PlusHDF':tracts_nhpialone['LessThan20MDFHundredPlusHDF'].sum()}))
outputdflist.append(ss)

# Counties Total Population Cross 50000
hdfcounties_totalpop = dfhdf().group_by('CountyGEOID').agg(HDF_Population = pl.len())
mdfcounties_totalpop = dfmdf().group_by('CountyGEOID').agg(MDF_Population = pl.len())
counties_totalpop =  hdfcounties_totalpop.join(mdfcounties_totalpop, on='GEOID', how="full", coalesce=True)
counties_totalpop = counties_totalpop.with_columns(pl.when((pl.col('MDF_Population') < 50000) & (pl.col('HDF_Population') > 50000)).then(1).otherwise(0).alias('HDFgt50kMDFlt50k'))
counties_totalpop = counties_totalpop.with_columns(pl.when((pl.col('MDF_Population') > 50000) & (pl.col('HDF_Population') < 50000)).then(1).otherwise(0).alias('HDFlt50kMDFgt50k'))
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'Cross 50000','NumCells': len(counties_totalpop),'NumberHDFgt50kMDFlt50k': counties_totalpop['HDFgt50kMDFlt50k'].sum(), 'NumberHDFlt50kMDFgt50k': counties_totalpop['HDFlt50kMDFgt50k'].sum()})
outputdflist.append(ss)

# Places Total Population Cross 50000
hdfplaces_totalpop = dfhdf().group_by(['IncPlaceGEOID']).agg(HDF_Population = pl.len())
mdfplaces_totalpop = dfmdf().group_by(['IncPlaceGEOID']).agg(MDF_Population = pl.len())
places_totalpop =  hdfplaces_totalpop.join(mdfplaces_totalpop, on='IncPlaceGEOID', how="full", coalesce=True)
places_totalpop = places_totalpop.with_columns(pl.when((pl.col('MDF_Population') < 50000) & (pl.col('HDF_Population') > 50000)).then(1).otherwise(0).alias('HDFgt50kMDFlt50k'))
places_totalpop = places_totalpop.with_columns(pl.when((pl.col('MDF_Population') > 50000) & (pl.col('HDF_Population') < 50000)).then(1).otherwise(0).alias('HDFlt50kMDFgt50k'))
ss = pl.DataFrame({'Geography':'Place', 'Size_Category':'All', 'Characteristic':'Cross 50000','NumCells': len(places_totalpop),'NumberHDFgt50kMDFlt50k': places_totalpop['HDFgt50kMDFlt50k'].sum(), 'NumberHDFlt50kMDFgt50k': places_totalpop['HDFlt50kMDFgt50k'].sum()})
outputdflist.append(ss)

# Tracts Total Population Cross 50000
hdftracts_totalpop = dfhdf().group_by('TractGEOID').agg(HDF_Population = pl.len())
mdftracts_totalpop = dfmdf().group_by('TractGEOID').agg(MDF_Population = pl.len())
tracts_totalpop =  hdftracts_totalpop.join(mdftracts_totalpop, on='GEOID', how="full", coalesce=True)
tracts_totalpop = tracts_totalpop.with_columns(pl.when((pl.col('MDF_Population') < 50000) & (pl.col('HDF_Population') > 50000)).then(1).otherwise(0).alias('HDFgt50kMDFlt50k'))
tracts_totalpop = tracts_totalpop.with_columns(pl.when((pl.col('MDF_Population') > 50000) & (pl.col('HDF_Population') < 50000)).then(1).otherwise(0).alias('HDFlt50kMDFgt50k'))
ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'Cross 50000','NumCells': len(tracts_totalpop),'NumberHDFgt50kMDFlt50k': tracts_totalpop['HDFgt50kMDFlt50k'].sum(), 'NumberHDFlt50kMDFgt50k': tracts_totalpop['HDFlt50kMDFgt50k'].sum()})
outputdflist.append(ss)


print("{} Use Cases Done".format(datetime.now()))

print("{} Starting Improbable and Impossible Measurements".format(datetime.now()))

# States Total Population Should Be Equal
hdfstates_totalpop = dfhdf().group_by(['TABBLKST']).agg(HDF_Population = pl.len())
mdfstates_totalpop = dfmdf().group_by(['TABBLKST']).agg(MDF_Population = pl.len())
states_totalpop =  hdfstates_totalpop.join(mdfstates_totalpop, on='GEOID', how="full", coalesce=True)
states_totalpop = states_totalpop.with_columns(pl.when(pl.col('MDF_Population') != pl.col('HDF_Population')).then(1).otherwise(0).alias('HDFneMDF'))
ss = pl.DataFrame({'Geography':'State', 'Size_Category':'All', 'Characteristic':'Total Population','NumCells': len(states_totalpop),'NumberHDFneMDF': states_totalpop['HDFneMDF'].sum()})
outputdflist.append(ss)

# Counties with at least 5 children under age 5 and no women age 18 through 44
hdfcounties_poplt5 = dfhdf().filter(pl.col('QAGE') < 5).group_by('CountyGEOID').agg(pl.len().alias('HDF_Children'))
mdfcounties_poplt5 = dfmdf().filter(pl.col('QAGE') < 5).group_by('CountyGEOID').agg(pl.len().alias('MDF_Children'))
hdfcounties_popfem1844 = dfhdf().filter((pl.col('QSEX') == '2'), (pl.col('QAGE') >= 18), (pl.col('QAGE') < 45)).group_by('CountyGEOID').agg(pl.len().alias('HDF_MomAge'))
mdfcounties_popfem1844 = dfmdf().filter((pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('CountyGEOID').agg(pl.len().alias('MDF_MomAge'))

hdfcounties_poplt5 = hdfcounties_poplt5.filter(pl.col('HDF_Children') >= 5)
mdfcounties_poplt5 = mdfcounties_poplt5.filter(pl.col('MDF_Children') >= 5)

hdfcounties =  hdfcounties_poplt5.join(hdfcounties_popfem1844, on='GEOID', how="left", coalesce=True)
mdfcounties =  mdfcounties_poplt5.join(mdfcounties_popfem1844, on='GEOID', how="left", coalesce=True)

hdfcounties = hdfcounties.with_columns(pl.when((pl.col('HDF_Children') >= 5)&(pl.col('HDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))
mdfcounties = mdfcounties.with_columns(pl.when((pl.col('MDF_Children') >= 5)&(pl.col('MDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))

ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'HDF Children No Moms','NumCells':len(hdfcounties), 'Inconsistent':hdfcounties['ChildrenNoMoms'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'MDF Children No Moms','NumCells':len(mdfcounties), 'Inconsistent':mdfcounties['ChildrenNoMoms'].sum()})
outputdflist.append(ss)

# Tracts with at least 5 children under age 5 and no women age 18 through 44
hdftracts_poplt5 = dfhdf().filter(pl.col('QAGE') < 5).group_by('TractGEOID').agg(pl.len().alias('HDF_Children')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
mdftracts_poplt5 = dfmdf().filter(pl.col('QAGE') < 5).group_by('TractGEOID').agg(pl.len().alias('MDF_Children')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
hdftracts_popfem1844 = dfhdf().filter((pl.col('QSEX') == '2'), (pl.col('QAGE') >= 18), (pl.col('QAGE') < 45)).group_by('TractGEOID').agg(pl.len().alias('HDF_MomAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
mdftracts_popfem1844 = dfmdf().filter((pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('TractGEOID').agg(pl.len().alias('MDF_MomAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)

hdftracts_poplt5 = hdftracts_poplt5.filter(pl.col('HDF_Children') >= 5)
mdftracts_poplt5 = mdftracts_poplt5.filter(pl.col('MDF_Children') >= 5)

hdftracts =  hdftracts_poplt5.join(hdftracts_popfem1844, on='GEOID', how="left", coalesce=True)
mdftracts =  mdftracts_poplt5.join(mdftracts_popfem1844, on='GEOID', how="left", coalesce=True)

hdftracts = hdftracts.with_columns(pl.when((pl.col('HDF_Children') >= 5)&(pl.col('HDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))
mdftracts = mdftracts.with_columns(pl.when((pl.col('MDF_Children') >= 5)&(pl.col('MDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))

ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF Children No Moms','NumCells':len(hdftracts), 'Inconsistent':hdftracts['ChildrenNoMoms'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF Children No Moms','NumCells':len(mdftracts), 'Inconsistent':mdftracts['ChildrenNoMoms'].sum()})
outputdflist.append(ss)

# Counties with at least 5 children under age 5 and no women age 18 through 44 by race alone
for r in racealonecats:
    hdfcounties_poplt5 = dfhdf().filter((pl.col('RACEALONE') == r)&(pl.col('QAGE') < 5).group_by('CountyGEOID').agg(pl.len().alias('HDF_Children'))
    mdfcounties_poplt5 = dfmdf().filter((pl.col('RACEALONE') == r)&(pl.col('QAGE') < 5).group_by('CountyGEOID').agg(pl.len().alias('MDF_Children'))
    hdfcounties_popfem1844 = dfhdf().filter((pl.col('RACEALONE') == r)&(pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('CountyGEOID').agg(pl.len().alias('HDF_MomAge'))
    mdfcounties_popfem1844 = dfmdf().filter((pl.col('RACEALONE') == r)&(pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('CountyGEOID').agg(pl.len().alias('MDF_MomAge'))

    hdfcounties_poplt5 = hdfcounties_poplt5.filter(pl.col('HDF_Children') >= 5)
    mdfcounties_poplt5 = mdfcounties_poplt5.filter(pl.col('MDF_Children') >= 5)

    hdfcounties =  hdfcounties_poplt5.join(hdfcounties_popfem1844, on='GEOID', how="left", coalesce=True)
    mdfcounties =  mdfcounties_poplt5.join(mdfcounties_popfem1844, on='GEOID', how="left", coalesce=True)

    hdfcounties = hdfcounties.with_columns(pl.when((pl.col('HDF_Children') >= 5)&(pl.col('HDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))
    mdfcounties = mdfcounties.with_columns(pl.when((pl.col('MDF_Children') >= 5)&(pl.col('MDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))

    ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"HDF Children No Moms {race}".format(race = racealonedict.get(r)),'NumCells':len(hdfcounties), 'Inconsistent':hdfcounties['ChildrenNoMoms'].sum()})
    outputdflist.append(ss)
    ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"MDF Children No Moms {race}".format(race = racealonedict.get(r)),'NumCells':len(mdfcounties), 'Inconsistent':mdfcounties['ChildrenNoMoms'].sum()})
    outputdflist.append(ss)

# Counties with at least 5 children under age 5 and no women age 18 through 44 Hispanic
hdfcounties_poplt5 = dfhdf().filter((pl.col('CENHISP') == '2')&(pl.col('QAGE') < 5).group_by('CountyGEOID').agg(pl.len().alias('HDF_Children'))
mdfcounties_poplt5 = dfmdf().filter((pl.col('CENHISP') == '2')&(pl.col('QAGE') < 5).group_by('CountyGEOID').agg(pl.len().alias('MDF_Children'))
hdfcounties_popfem1844 = dfhdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('CountyGEOID').agg(pl.len().alias('HDF_MomAge'))
mdfcounties_popfem1844 = dfmdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('CountyGEOID').agg(pl.len().alias('MDF_MomAge'))

hdfcounties_poplt5 = hdfcounties_poplt5.filter(pl.col('HDF_Children') >= 5)
mdfcounties_poplt5 = mdfcounties_poplt5.filter(pl.col('MDF_Children') >= 5)

hdfcounties =  hdfcounties_poplt5.join(hdfcounties_popfem1844, on='GEOID', how="left", coalesce=True)
mdfcounties =  mdfcounties_poplt5.join(mdfcounties_popfem1844, on='GEOID', how="left", coalesce=True)

hdfcounties = hdfcounties.with_columns(pl.when((pl.col('HDF_Children') >= 5)&(pl.col('HDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))
mdfcounties = mdfcounties.with_columns(pl.when((pl.col('MDF_Children') >= 5)&(pl.col('MDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))

ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'HDF Children No Moms Hispanic','NumCells':len(hdfcounties), 'Inconsistent':hdfcounties['ChildrenNoMoms'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'MDF Children No Moms Hispanic','NumCells':len(mdfcounties), 'Inconsistent':mdfcounties['ChildrenNoMoms'].sum()})
outputdflist.append(ss)

# Counties with at least 5 children under age 5 and no women age 18 through 44 Non-Hispanic White
hdfcounties_poplt5 = dfhdf().filter((pl.col('CENHISP') == '1')&(pl.col('RACEALONE') == 1)&(pl.col('QAGE') < 5).group_by('CountyGEOID').agg(pl.len().alias('HDF_Children'))
mdfcounties_poplt5 = dfmdf().filter((pl.col('CENHISP') == '1')&(pl.col('RACEALONE') == 1)&(pl.col('QAGE') < 5).group_by('CountyGEOID').agg(pl.len().alias('MDF_Children'))
hdfcounties_popfem1844 = dfhdf().filter((pl.col('CENHISP') == '1')&(pl.col('RACEALONE') == 1)&(pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('CountyGEOID').agg(pl.len().alias('HDF_MomAge'))
mdfcounties_popfem1844 = dfmdf().filter((pl.col('CENHISP') == '1')&(pl.col('RACEALONE') == 1)&(pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('CountyGEOID').agg(pl.len().alias('MDF_MomAge'))

hdfcounties_poplt5 = hdfcounties_poplt5.filter(pl.col('HDF_Children') >= 5)
mdfcounties_poplt5 = mdfcounties_poplt5.filter(pl.col('MDF_Children') >= 5)

hdfcounties =  hdfcounties_poplt5.join(hdfcounties_popfem1844, on='GEOID', how="left", coalesce=True)
mdfcounties =  mdfcounties_poplt5.join(mdfcounties_popfem1844, on='GEOID', how="left", coalesce=True)

hdfcounties = hdfcounties.with_columns(pl.when((pl.col('HDF_Children') >= 5)&(pl.col('HDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))
mdfcounties = mdfcounties.with_columns(pl.when((pl.col('MDF_Children') >= 5)&(pl.col('MDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))

ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'HDF Children No Moms Non-Hispanic White','NumCells':len(hdfcounties), 'Inconsistent':hdfcounties['ChildrenNoMoms'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'MDF Children No Moms Non-Hispanic White','NumCells':len(mdfcounties), 'Inconsistent':mdfcounties['ChildrenNoMoms'].sum()})
outputdflist.append(ss)

# Tracts with at least 5 children under age 5 and no women age 18 through 44 by race alone
for r in racealonecats:
    hdftracts_poplt5 = dfhdf().filter((pl.col('RACEALONE') == r)&(pl.col('QAGE') < 5).group_by('TractGEOID').agg(pl.len().alias('HDF_Children')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
    mdftracts_poplt5 = dfmdf().filter((pl.col('RACEALONE') == r)&(pl.col('QAGE') < 5).group_by('TractGEOID').agg(pl.len().alias('MDF_Children')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
    hdftracts_popfem1844 = dfhdf().filter((pl.col('RACEALONE') == r)&(pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('TractGEOID').agg(pl.len().alias('HDF_MomAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
    mdftracts_popfem1844 = dfmdf().filter((pl.col('RACEALONE') == r)&(pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('TractGEOID').agg(pl.len().alias('MDF_MomAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
    
    hdftracts_poplt5 = hdftracts_poplt5.filter(pl.col('HDF_Children') >= 5)
    mdftracts_poplt5 = mdftracts_poplt5.filter(pl.col('MDF_Children') >= 5)

    hdftracts =  hdftracts_poplt5.join(hdftracts_popfem1844, on='GEOID', how="left", coalesce=True)
    mdftracts =  mdftracts_poplt5.join(mdftracts_popfem1844, on='GEOID', how="left", coalesce=True)

    hdftracts = hdftracts.with_columns(pl.when((pl.col('HDF_Children') >= 5)&(pl.col('HDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))
    mdftracts = mdftracts.with_columns(pl.when((pl.col('MDF_Children') >= 5)&(pl.col('MDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))

    ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':"HDF Children No Moms {race}".format(race = racealonedict.get(r)),'NumCells':len(hdftracts), 'Inconsistent':hdftracts['ChildrenNoMoms'].sum()})
    outputdflist.append(ss)
    ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':"MDF Children No Moms {race}".format(race = racealonedict.get(r)),'NumCells':len(mdftracts), 'Inconsistent':mdftracts['ChildrenNoMoms'].sum()})
    outputdflist.append(ss)

# Tracts with at least 5 children under age 5 and no women age 18 through 44 Hispanic
hdftracts_poplt5 = dfhdf().filter((pl.col('CENHISP') == '2')&(pl.col('QAGE') < 5).group_by('TractGEOID').agg(pl.len().alias('HDF_Children')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
mdftracts_poplt5 = dfmdf().filter((pl.col('CENHISP') == '2')&(pl.col('QAGE') < 5).group_by('TractGEOID').agg(pl.len().alias('MDF_Children')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
hdftracts_popfem1844 = dfhdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('TractGEOID').agg(pl.len().alias('HDF_MomAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
mdftracts_popfem1844 = dfmdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('TractGEOID').agg(pl.len().alias('MDF_MomAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)

hdftracts_poplt5 = hdftracts_poplt5.filter(pl.col('HDF_Children') >= 5)
mdftracts_poplt5 = mdftracts_poplt5.filter(pl.col('MDF_Children') >= 5)

hdftracts =  hdftracts_poplt5.join(hdftracts_popfem1844, on='GEOID', how="left", coalesce=True)
mdftracts =  mdftracts_poplt5.join(mdftracts_popfem1844, on='GEOID', how="left", coalesce=True)

hdftracts = hdftracts.with_columns(pl.when((pl.col('HDF_Children') >= 5)&(pl.col('HDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))
mdftracts = mdftracts.with_columns(pl.when((pl.col('MDF_Children') >= 5)&(pl.col('MDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))

ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF Children No Moms Hispanic','NumCells':len(hdftracts), 'Inconsistent':hdftracts['ChildrenNoMoms'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF Children No Moms Hispanic','NumCells':len(mdftracts), 'Inconsistent':mdftracts['ChildrenNoMoms'].sum()})
outputdflist.append(ss)

# Tracts with at least 5 children under age 5 and no women age 18 through 44 Non-Hispanic White
hdftracts_poplt5 = dfhdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QAGE') < 5).group_by('TractGEOID').agg(pl.len().alias('HDF_Children')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
mdftracts_poplt5 = dfmdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QAGE') < 5).group_by('TractGEOID').agg(pl.len().alias('MDF_Children')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
hdftracts_popfem1844 = dfhdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('TractGEOID').agg(pl.len().alias('HDF_MomAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
mdftracts_popfem1844 = dfmdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '2')&(pl.col('QAGE') >= 18)&(pl.col('QAGE') < 45).group_by('TractGEOID').agg(pl.len().alias('MDF_MomAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)

hdftracts_poplt5 = hdftracts_poplt5.filter(pl.col('HDF_Children') >= 5)
mdftracts_poplt5 = mdftracts_poplt5.filter(pl.col('MDF_Children') >= 5)

hdftracts =  hdftracts_poplt5.join(hdftracts_popfem1844, on='GEOID', how="left", coalesce=True)
mdftracts =  mdftracts_poplt5.join(mdftracts_popfem1844, on='GEOID', how="left", coalesce=True)

hdftracts = hdftracts.with_columns(pl.when((pl.col('HDF_Children') >= 5)&(pl.col('HDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))
mdftracts = mdftracts.with_columns(pl.when((pl.col('MDF_Children') >= 5)&(pl.col('MDF_MomAge') == 0)).then(1).otherwise(0).alias('ChildrenNoMoms'))

ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF Children No Moms Non-Hispanic White','NumCells':len(hdftracts), 'Inconsistent':hdftracts['ChildrenNoMoms'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF Children No Moms Non-Hispanic White','NumCells':len(mdftracts), 'Inconsistent':mdftracts['ChildrenNoMoms'].sum()})
outputdflist.append(ss)

# Tracts with at least 5 people and all of the same sex
hdftracts_males = dfhdf().filter(pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.len().alias('HDF_Males')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
mdftracts_males = dfmdf().filter(pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.len().alias('MDF_Males')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
hdftracts_females = dfhdf().filter(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.len().alias('HDF_Females')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
mdftracts_females = dfmdf().filter(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.len().alias('MDF_Females')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)

hdftracts =  hdftracts_males.join(hdftracts_females, on='GEOID', how="full", coalesce=True)
mdftracts =  mdftracts_males.join(mdftracts_females, on='GEOID', how="full", coalesce=True)

hdftracts = hdftracts.with_columns((pl.col('HDF_Females') + pl.col('HDF_Males')).alias('Total'))
mdftracts = mdftracts.with_columns((pl.col('MDF_Females') + pl.col('MDF_Males')).alias('Total'))

hdftracts = hdftracts.filter(pl.col('Total') >= 5)
mdftracts = mdftracts.filter(pl.col('Total') >= 5)

hdftracts = hdftracts.with_columns(pl.when((pl.col('HDF_Males') == 0)|(pl.col('HDF_Females') == 0)).then(1).otherwise(0).alias('AllSameSex'))
mdftracts = mdftracts.with_columns(pl.when((pl.col('MDF_Males') == 0)|(pl.col('MDF_Females') == 0)).then(1).otherwise(0).alias('AllSameSex'))

ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF All Same Sex','NumCells':len(hdftracts), 'Inconsistent':hdftracts['AllSameSex'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF All Same Sex','NumCells':len(mdftracts), 'Inconsistent':mdftracts['AllSameSex'].sum()})
outputdflist.append(ss)

# Tracts with at least one of the single years of age between 0 and 17 by sex has a zero count
hdftracts_totunder18 = dfhdf().filter(pl.col('QAGE') < 18).group_by('TractGEOID').agg(pl.len().alias('HDF_TotUnder18')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
mdftracts_totunder18 = dfmdf().filter(pl.col('QAGE') < 18).group_by('TractGEOID').agg(pl.len().alias('MDF_TotUnder18')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': alltractsindex}), on='GEOID', how='right', coalesce=True).fill_null(0)

hdftracts_totunder18gt200 = hdftracts_totunder18.filter(pl.col('HDF_TotUnder18') > 200).get_column('GEOID').to_list()
mdftracts_totunder18gt200 = mdftracts_totunder18.filter(pl.col('MDF_TotUnder18') > 200).get_column('GEOID').to_list()

# hdftract_1yageunder18_index = MultiIndex (handled via join below)
# mdftract_1yageunder18_index = MultiIndex (handled via join below)

hdftracts_under18 = dfhdf().filter(pl.col('QAGE') < 18).group_by(['TractGEOID', 'QAGE']).agg(HDF_Population = pl.len())
mdftracts_under18 = dfmdf().filter(pl.col('QAGE') < 18).group_by(['TractGEOID', 'QAGE']).agg(MDF_Population = pl.len())

hdftracts_under18 = hdftracts_under18.with_columns(pl.when((pl.col('HDF_Population') == 0)).then(1).otherwise(0).alias('ZeroAge'))
mdftracts_under18 = mdftracts_under18.with_columns(pl.when((pl.col('MDF_Population') == 0)).then(1).otherwise(0).alias('ZeroAge'))

hdftracts_anyzeros = hdftracts_under18.group_by('TractGEOID').agg(pl.col('ZeroAge').max())
mdftracts_anyzeros = mdftracts_under18.group_by('TractGEOID').agg(pl.col('ZeroAge').max())

ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF Zero Age','NumCells':len(hdftracts_anyzeros), 'Inconsistent':hdftracts_anyzeros['ZeroAge'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF Zero Age','NumCells':len(mdftracts_anyzeros), 'Inconsistent':mdftracts_anyzeros['ZeroAge'].sum()})
outputdflist.append(ss)

hdftracts_under18.write_csv(f"{OUTPUTDIR}/hdftracts_under18.csv")
hdftracts_anyzeros.write_csv(f"{OUTPUTDIR}/hdftracts_under18anyzeros.csv")

# Blocks with population all 17 or younger 
hdfblocks_gqpop = dfhdf().filter(pl.col('RTYPE') == "5").group_by('BlockGEOID').agg(pl.len().alias('HDF_GQPopulation')).join(pl.DataFrame({'GEOID': allblocksindex}), on='GEOID', how='right', coalesce=True).fill_null(0)
mdfblocks_gqpop = dfmdf().filter(pl.col('RTYPE') == "5").group_by('BlockGEOID').agg(pl.len().alias('MDF_GQPopulation')).join(pl.DataFrame({'GEOID': allblocksindex}), on='GEOID', how='right', coalesce=True).fill_null(0)

hdfblocks_nogqs = hdfblocks_gqpop.filter(pl.col('HDF_GQPopulation') == 0).get_column('GEOID').to_list()
mdfblocks_nogqs = mdfblocks_gqpop.filter(pl.col('MDF_GQPopulation') == 0).get_column('GEOID').to_list()

del mdfblocks_gqpop
del hdfblocks_gqpop

hdfblocks_allpop = dfhdf().group_by(['TABBLKST','TABBLKCOU','TABTRACTCE', 'TABBLK']).agg(HDF_Population = pl.len())
mdfblocks_allpop = dfmdf().group_by(['TABBLKST','TABBLKCOU','TABTRACTCE', 'TABBLK']).agg(MDF_Population = pl.len())

hdfblocks_somepop = hdfblocks_allpop.filter(pl.col('HDF_Population') > 0).get_column('GEOID').to_list()
mdfblocks_somepop = mdfblocks_allpop.filter(pl.col('MDF_Population') > 0).get_column('GEOID').to_list()

del hdfblocks_allpop
del mdfblocks_allpop

hdfblocks_nogqs_somepop = set(hdfblocks_nogqs).intersection(hdfblocks_somepop)
mdfblocks_nogqs_somepop = set(mdfblocks_nogqs).intersection(mdfblocks_somepop)

hdfblocks_nogqs_index = hdfblocks_nogqs_somepop
mdfblocks_nogqs_index = mdfblocks_nogqs_somepop

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

hdfblocks_18 = dfhdf().filter(pl.col('QAGE') < 18).group_by('BlockGEOID').agg(pl.len().alias('HDF_Under18')).join(pl.DataFrame({'BlockGEOID': hdfblocks_nogqs_index}), on='BlockGEOID', how='right', coalesce=True).fill_null(0)
mdfblocks_18 = dfmdf().filter(pl.col('QAGE') < 18).group_by('BlockGEOID').agg(pl.len().alias('MDF_Under18')).join(pl.DataFrame({'BlockGEOID': mdfblocks_nogqs_index}), on='BlockGEOID', how='right', coalesce=True).fill_null(0)

hdfblocks =  hdfblocks_totpop.join(hdfblocks_18, on='BlockGEOID', how="inner", coalesce=True)
mdfblocks =  mdfblocks_totpop.join(mdfblocks_18, on='BlockGEOID', how="inner", coalesce=True)

del hdfblocks_18
del mdfblocks_18
del hdfblocks_totpop
del mdfblocks_totpop

hdfblocks = hdfblocks.with_columns(pl.when((pl.col('HDF_Population') > 0)&(pl.col('HDF_Under18') == pl.col('HDF_Population'))).then(1).otherwise(0).alias('Zero18andOver'))
mdfblocks = mdfblocks.with_columns(pl.when((pl.col('MDF_Population') > 0)&(pl.col('MDF_Under18') == pl.col('MDF_Population'))).then(1).otherwise(0).alias('Zero18andOver'))

ss = pl.DataFrame({'Geography':'Block', 'Size_Category':'All', 'Characteristic':'HDF Everyone Under 18','NumCells':len(hdfblocks), 'Inconsistent':hdfblocks['Zero18andOver'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'Block', 'Size_Category':'All', 'Characteristic':'MDF Everyone Under 18','NumCells':len(mdfblocks), 'Inconsistent':mdfblocks['Zero18andOver'].sum()})
outputdflist.append(ss)

del hdfblocks
del mdfblocks
del hdfblocks_nogqs_index
del mdfblocks_nogqs_index


# Counties where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women
hdfcounties_males = dfhdf().filter(pl.col('QSEX') == '1').group_by('CountyGEOID').agg(pl.len().alias('HDF_Males'))
mdfcounties_males = dfmdf().filter(pl.col('QSEX') == '1').group_by('CountyGEOID').agg(pl.len().alias('MDF_Males'))
hdfcounties_females = dfhdf().filter(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.len().alias('HDF_Females'))
mdfcounties_females = dfmdf().filter(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.len().alias('MDF_Females'))

hdfcounties_gt5males = hdfcounties_males.filter(pl.col('HDF_Males') >= 5).get_column('GEOID').to_list()
mdfcounties_gt5males = mdfcounties_males.filter(pl.col('MDF_Males') >= 5).get_column('GEOID').to_list()
hdfcounties_gt5females = hdfcounties_females.filter(pl.col('HDF_Females') >= 5).get_column('GEOID').to_list()
mdfcounties_gt5females = mdfcounties_females.filter(pl.col('MDF_Females') >= 5).get_column('GEOID').to_list()

hdfcounties_gt5bothsex =  list(set(hdfcounties_gt5males).intersection(hdfcounties_gt5females))
mdfcounties_gt5bothsex =  list(set(mdfcounties_gt5males).intersection(mdfcounties_gt5females))

hdfcounties_gt5bothsex_index = hdfcounties_gt5bothsex
mdfcounties_gt5bothsex_index = mdfcounties_gt5bothsex

hdfcounties_malemedage = dfhdf().filter(pl.col('QSEX') == '1').group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_MaleAge')).collect().join(pl.DataFrame({'GEOID': hdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
mdfcounties_malemedage = dfmdf().filter(pl.col('QSEX') == '1').group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_MaleAge')).collect().join(pl.DataFrame({'GEOID': mdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
hdfcounties_femalemedage = dfhdf().filter(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_FemaleAge')).collect().join(pl.DataFrame({'GEOID': hdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
mdfcounties_femalemedage = dfmdf().filter(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_FemaleAge')).collect().join(pl.DataFrame({'GEOID': mdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)

hdfcounties =  hdfcounties_malemedage.join(hdfcounties_femalemedage, on='GEOID', how="inner", coalesce=True)
mdfcounties =  mdfcounties_malemedage.join(mdfcounties_femalemedage, on='GEOID', how="inner", coalesce=True)

hdfcounties = hdfcounties.with_columns(pl.when((pl.col('HDF_MaleAge').abs() - pl.col('HDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))
mdfcounties = mdfcounties.with_columns(pl.when((pl.col('MDF_MaleAge').abs() - pl.col('MDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))

ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'HDF 20+ Year Median Age Diff','NumCells':len(hdfcounties), 'Inconsistent':hdfcounties['BigAgeDiff'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':'MDF 20+ Year Median Age Diff','NumCells':len(mdfcounties), 'Inconsistent':mdfcounties['BigAgeDiff'].sum()})
outputdflist.append(ss)

# Counties where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women, by major race group
for r in racealonecats:
    hdfcounties_males = dfhdf().filter((pl.col('RACEALONE') == r), (pl.col('QSEX') == '1').group_by('CountyGEOID').agg(pl.len().alias('HDF_Males'))
    mdfcounties_males = dfmdf().filter((pl.col('RACEALONE') == r), (pl.col('QSEX') == '1')).group_by('CountyGEOID').agg(pl.len().alias('MDF_Males'))
    hdfcounties_females = dfhdf().filter((pl.col('RACEALONE') == r)&(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.len().alias('HDF_Females'))
    mdfcounties_females = dfmdf().filter((pl.col('RACEALONE') == r)&(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.len().alias('MDF_Females'))

    hdfcounties_gt5males = hdfcounties_males.filter(pl.col('HDF_Males') >= 5).get_column('GEOID').to_list()
    mdfcounties_gt5males = mdfcounties_males.filter(pl.col('MDF_Males') >= 5).get_column('GEOID').to_list()
    hdfcounties_gt5females = hdfcounties_females.filter(pl.col('HDF_Females') >= 5).get_column('GEOID').to_list()
    mdfcounties_gt5females = mdfcounties_females.filter(pl.col('MDF_Females') >= 5).get_column('GEOID').to_list()

    hdfcounties_gt5bothsex =  list(set(hdfcounties_gt5males).intersection(hdfcounties_gt5females))
    mdfcounties_gt5bothsex =  list(set(mdfcounties_gt5males).intersection(mdfcounties_gt5females))

    hdfcounties_gt5bothsex_index = hdfcounties_gt5bothsex
    mdfcounties_gt5bothsex_index = mdfcounties_gt5bothsex

    hdfcounties_malemedage = dfhdf().filter((pl.col('RACEALONE') == r), (pl.col('QSEX') == '1').group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_MaleAge')).collect().join(pl.DataFrame({'GEOID': hdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
    mdfcounties_malemedage = dfmdf().filter((pl.col('RACEALONE') == r), (pl.col('QSEX') == '1')).group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_MaleAge')).collect().join(pl.DataFrame({'GEOID': mdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
    hdfcounties_femalemedage = dfhdf().filter((pl.col('RACEALONE') == r)&(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_FemaleAge')).collect().join(pl.DataFrame({'GEOID': hdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
    mdfcounties_femalemedage = dfmdf().filter((pl.col('RACEALONE') == r)&(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_FemaleAge')).collect().join(pl.DataFrame({'GEOID': mdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)

    hdfcounties =  hdfcounties_malemedage.join(hdfcounties_femalemedage, on='GEOID', how="inner", coalesce=True)
    mdfcounties =  mdfcounties_malemedage.join(mdfcounties_femalemedage, on='GEOID', how="inner", coalesce=True)

    hdfcounties = hdfcounties.with_columns(pl.when((pl.col('HDF_MaleAge').abs() - pl.col('HDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))
    mdfcounties = mdfcounties.with_columns(pl.when((pl.col('MDF_MaleAge').abs() - pl.col('MDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))

    ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"HDF 20+ Year Median Age Diff {race}".format(race = racealonedict.get(r)),'NumCells':len(hdfcounties), 'Inconsistent':hdfcounties['BigAgeDiff'].sum()})
    outputdflist.append(ss)
    ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"MDF 20+ Year Median Age Diff {race}".format(race = racealonedict.get(r)),'NumCells':len(mdfcounties), 'Inconsistent':mdfcounties['BigAgeDiff'].sum()})
    outputdflist.append(ss)

# Counties where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women Hispanic
hdfcounties_males = dfhdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '1').group_by('CountyGEOID').agg(pl.len().alias('HDF_Males'))
mdfcounties_males = dfmdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '1').group_by('CountyGEOID').agg(pl.len().alias('MDF_Males'))
hdfcounties_females = dfhdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.len().alias('HDF_Females'))
mdfcounties_females = dfmdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.len().alias('MDF_Females'))

hdfcounties_gt5males = hdfcounties_males.filter(pl.col('HDF_Males') >= 5).get_column('GEOID').to_list()
mdfcounties_gt5males = mdfcounties_males.filter(pl.col('MDF_Males') >= 5).get_column('GEOID').to_list()
hdfcounties_gt5females = hdfcounties_females.filter(pl.col('HDF_Females') >= 5).get_column('GEOID').to_list()
mdfcounties_gt5females = mdfcounties_females.filter(pl.col('MDF_Females') >= 5).get_column('GEOID').to_list()

hdfcounties_gt5bothsex =  list(set(hdfcounties_gt5males).intersection(hdfcounties_gt5females))
mdfcounties_gt5bothsex =  list(set(mdfcounties_gt5males).intersection(mdfcounties_gt5females))

hdfcounties_gt5bothsex_index = hdfcounties_gt5bothsex
mdfcounties_gt5bothsex_index = mdfcounties_gt5bothsex

hdfcounties_malemedage = dfhdf().filter((pl.col('CENHISP') == '2'), (pl.col('QSEX') == '1').group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_MaleAge')).collect().join(pl.DataFrame({'GEOID': hdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
mdfcounties_malemedage = dfmdf().filter((pl.col('CENHISP') == '2'), (pl.col('QSEX') == '1')).group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_MaleAge')).collect().join(pl.DataFrame({'GEOID': mdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
hdfcounties_femalemedage = dfhdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_FemaleAge')).collect().join(pl.DataFrame({'GEOID': hdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
mdfcounties_femalemedage = dfmdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_FemaleAge')).collect().join(pl.DataFrame({'GEOID': mdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)

hdfcounties =  hdfcounties_malemedage.join(hdfcounties_femalemedage, on='GEOID', how="inner", coalesce=True)
mdfcounties =  mdfcounties_malemedage.join(mdfcounties_femalemedage, on='GEOID', how="inner", coalesce=True)

hdfcounties = hdfcounties.with_columns(pl.when((pl.col('HDF_MaleAge').abs() - pl.col('HDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))
mdfcounties = mdfcounties.with_columns(pl.when((pl.col('MDF_MaleAge').abs() - pl.col('MDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))

ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"HDF 20+ Year Median Age Diff Hispanic",'NumCells':len(hdfcounties), 'Inconsistent':hdfcounties['BigAgeDiff'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"MDF 20+ Year Median Age Diff Hispanic",'NumCells':len(mdfcounties), 'Inconsistent':mdfcounties['BigAgeDiff'].sum()})
outputdflist.append(ss)

# Counties where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women, Non-Hispanic White
hdfcounties_males = dfhdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '1').group_by('CountyGEOID').agg(pl.len().alias('HDF_Males'))
mdfcounties_males = dfmdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '1').group_by('CountyGEOID').agg(pl.len().alias('MDF_Males'))
hdfcounties_females = dfhdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.len().alias('HDF_Females'))
mdfcounties_females = dfmdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.len().alias('MDF_Females'))

hdfcounties_gt5males = hdfcounties_males.filter(pl.col('HDF_Males') >= 5).get_column('GEOID').to_list()
mdfcounties_gt5males = mdfcounties_males.filter(pl.col('MDF_Males') >= 5).get_column('GEOID').to_list()
hdfcounties_gt5females = hdfcounties_females.filter(pl.col('HDF_Females') >= 5).get_column('GEOID').to_list()
mdfcounties_gt5females = mdfcounties_females.filter(pl.col('MDF_Females') >= 5).get_column('GEOID').to_list()

hdfcounties_gt5bothsex =  list(set(hdfcounties_gt5males).intersection(hdfcounties_gt5females))
mdfcounties_gt5bothsex =  list(set(mdfcounties_gt5males).intersection(mdfcounties_gt5females))

hdfcounties_gt5bothsex_index = hdfcounties_gt5bothsex
mdfcounties_gt5bothsex_index = mdfcounties_gt5bothsex

hdfcounties_malemedage = dfhdf().filter((pl.col('RACEALONE') == 1), (pl.col('CENHISP') == '1'), (pl.col('QSEX') == '1')).group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_MaleAge')).collect().join(pl.DataFrame({'GEOID': hdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
mdfcounties_malemedage = dfmdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '1').group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_MaleAge')).collect().join(pl.DataFrame({'GEOID': mdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
hdfcounties_femalemedage = dfhdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_FemaleAge')).collect().join(pl.DataFrame({'GEOID': hdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
mdfcounties_femalemedage = dfmdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '2').group_by('CountyGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_FemaleAge')).collect().join(pl.DataFrame({'GEOID': mdfcounties_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)

hdfcounties =  hdfcounties_malemedage.join(hdfcounties_femalemedage, on='GEOID', how="inner", coalesce=True)
mdfcounties =  mdfcounties_malemedage.join(mdfcounties_femalemedage, on='GEOID', how="inner", coalesce=True)

hdfcounties = hdfcounties.with_columns(pl.when((pl.col('HDF_MaleAge').abs() - pl.col('HDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))
mdfcounties = mdfcounties.with_columns(pl.when((pl.col('MDF_MaleAge').abs() - pl.col('MDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))

ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"HDF 20+ Year Median Age Diff Non-Hispanic White",'NumCells':len(hdfcounties), 'Inconsistent':hdfcounties['BigAgeDiff'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'County', 'Size_Category':'All', 'Characteristic':"MDF 20+ Year Median Age Diff Non-Hispanic White",'NumCells':len(mdfcounties), 'Inconsistent':mdfcounties['BigAgeDiff'].sum()})
outputdflist.append(ss)

# Tracts where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women
hdftracts_males = dfhdf().filter(pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.len().alias('HDF_Males'))
mdftracts_males = dfmdf().filter(pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.len().alias('MDF_Males'))
hdftracts_females = dfhdf().filter(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.len().alias('HDF_Females'))
mdftracts_females = dfmdf().filter(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.len().alias('MDF_Females'))

hdftracts_gt5males = hdftracts_males.filter(pl.col('HDF_Males') >= 5).get_column('GEOID').to_list()
mdftracts_gt5males = mdftracts_males.filter(pl.col('MDF_Males') >= 5).get_column('GEOID').to_list()
hdftracts_gt5females = hdftracts_females.filter(pl.col('HDF_Females') >= 5).get_column('GEOID').to_list()
mdftracts_gt5females = mdftracts_females.filter(pl.col('MDF_Females') >= 5).get_column('GEOID').to_list()

hdftracts_gt5bothsex =  list(set(hdftracts_gt5males).intersection(hdftracts_gt5females))
mdftracts_gt5bothsex =  list(set(mdftracts_gt5males).intersection(mdftracts_gt5females))

hdftracts_gt5bothsex_index = hdftracts_gt5bothsex
mdftracts_gt5bothsex_index = mdftracts_gt5bothsex

hdftracts_malemedage = dfhdf().filter(pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_MaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': hdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
mdftracts_malemedage = dfmdf().filter(pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_MaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': mdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
hdftracts_femalemedage = dfhdf().filter(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_FemaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': hdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
mdftracts_femalemedage = dfmdf().filter(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_FemaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': mdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)

hdftracts =  hdftracts_malemedage.join(hdftracts_femalemedage, on='GEOID', how="inner", coalesce=True)
mdftracts =  mdftracts_malemedage.join(mdftracts_femalemedage, on='GEOID', how="inner", coalesce=True)

hdftracts = hdftracts.with_columns(pl.when((pl.col('HDF_MaleAge').abs() - pl.col('HDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))
mdftracts = mdftracts.with_columns(pl.when((pl.col('MDF_MaleAge').abs() - pl.col('MDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))

ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF 20+ Year Median Age Diff','NumCells':len(hdftracts), 'Inconsistent':hdftracts['BigAgeDiff'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF 20+ Year Median Age Diff','NumCells':len(mdftracts), 'Inconsistent':mdftracts['BigAgeDiff'].sum()})
outputdflist.append(ss)

# Tracts where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women by major race group
for r in racealonecats:
    hdftracts_males = dfhdf().filter((pl.col('RACEALONE') == r), (pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.len().alias('HDF_Males'))
    mdftracts_males = dfmdf().filter((pl.col('RACEALONE') == r), (pl.col('QSEX') == '1')).group_by('TractGEOID').agg(pl.len().alias('MDF_Males'))
    hdftracts_females = dfhdf().filter((pl.col('RACEALONE') == r)&(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.len().alias('HDF_Females'))
    mdftracts_females = dfmdf().filter((pl.col('RACEALONE') == r)&(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.len().alias('MDF_Females'))

    hdftracts_gt5males = hdftracts_males.filter(pl.col('HDF_Males') >= 5).get_column('GEOID').to_list()
    mdftracts_gt5males = mdftracts_males.filter(pl.col('MDF_Males') >= 5).get_column('GEOID').to_list()
    hdftracts_gt5females = hdftracts_females.filter(pl.col('HDF_Females') >= 5).get_column('GEOID').to_list()
    mdftracts_gt5females = mdftracts_females.filter(pl.col('MDF_Females') >= 5).get_column('GEOID').to_list()

    hdftracts_gt5bothsex =  list(set(hdftracts_gt5males).intersection(hdftracts_gt5females))
    mdftracts_gt5bothsex =  list(set(mdftracts_gt5males).intersection(mdftracts_gt5females))

    hdftracts_gt5bothsex_index = hdftracts_gt5bothsex
    mdftracts_gt5bothsex_index = mdftracts_gt5bothsex

    hdftracts_malemedage = dfhdf().filter((pl.col('RACEALONE') == r), (pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_MaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': hdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
    mdftracts_malemedage = dfmdf().filter((pl.col('RACEALONE') == r), (pl.col('QSEX') == '1')).group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_MaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': mdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
    hdftracts_femalemedage = dfhdf().filter((pl.col('RACEALONE') == r)&(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_FemaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': hdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
    mdftracts_femalemedage = dfmdf().filter((pl.col('RACEALONE') == r)&(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_FemaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': mdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)

    hdftracts =  hdftracts_malemedage.join(hdftracts_femalemedage, on='GEOID', how="inner", coalesce=True)
    mdftracts =  mdftracts_malemedage.join(mdftracts_femalemedage, on='GEOID', how="inner", coalesce=True)

    hdftracts = hdftracts.with_columns(pl.when((pl.col('HDF_MaleAge').abs() - pl.col('HDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))
    mdftracts = mdftracts.with_columns(pl.when((pl.col('MDF_MaleAge').abs() - pl.col('MDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))

    ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':"HDF 20+ Year Median Age Diff {race}".format(race = racealonedict.get(r)),'NumCells':len(hdftracts), 'Inconsistent':hdftracts['BigAgeDiff'].sum()})
    outputdflist.append(ss)
    ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':"MDF 20+ Year Median Age Diff {race}".format(race = racealonedict.get(r)),'NumCells':len(mdftracts), 'Inconsistent':mdftracts['BigAgeDiff'].sum()})
    outputdflist.append(ss)


# Tracts where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women Hispanic
hdftracts_males = dfhdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.len().alias('HDF_Males'))
mdftracts_males = dfmdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.len().alias('MDF_Males'))
hdftracts_females = dfhdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.len().alias('HDF_Females'))
mdftracts_females = dfmdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.len().alias('MDF_Females'))

hdftracts_gt5males = hdftracts_males.filter(pl.col('HDF_Males') >= 5).get_column('GEOID').to_list()
mdftracts_gt5males = mdftracts_males.filter(pl.col('MDF_Males') >= 5).get_column('GEOID').to_list()
hdftracts_gt5females = hdftracts_females.filter(pl.col('HDF_Females') >= 5).get_column('GEOID').to_list()
mdftracts_gt5females = mdftracts_females.filter(pl.col('MDF_Females') >= 5).get_column('GEOID').to_list()

hdftracts_gt5bothsex =  list(set(hdftracts_gt5males).intersection(hdftracts_gt5females))
mdftracts_gt5bothsex =  list(set(mdftracts_gt5males).intersection(mdftracts_gt5females))

hdftracts_gt5bothsex_index = hdftracts_gt5bothsex
mdftracts_gt5bothsex_index = mdftracts_gt5bothsex

hdftracts_malemedage = dfhdf().filter((pl.col('CENHISP') == '2'), (pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_MaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': hdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
mdftracts_malemedage = dfmdf().filter((pl.col('CENHISP') == '2'), (pl.col('QSEX') == '1')).group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_MaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': mdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
hdftracts_femalemedage = dfhdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_FemaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': hdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
mdftracts_femalemedage = dfmdf().filter((pl.col('CENHISP') == '2')&(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_FemaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': mdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)

hdftracts =  hdftracts_malemedage.join(hdftracts_femalemedage, on='GEOID', how="inner", coalesce=True)
mdftracts =  mdftracts_malemedage.join(mdftracts_femalemedage, on='GEOID', how="inner", coalesce=True)

hdftracts = hdftracts.with_columns(pl.when((pl.col('HDF_MaleAge').abs() - pl.col('HDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))
mdftracts = mdftracts.with_columns(pl.when((pl.col('MDF_MaleAge').abs() - pl.col('MDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))

ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF 20+ Year Median Age Diff Hispanic','NumCells':len(hdftracts), 'Inconsistent':hdftracts['BigAgeDiff'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF 20+ Year Median Age Diff Hispanic','NumCells':len(mdftracts), 'Inconsistent':mdftracts['BigAgeDiff'].sum()})
outputdflist.append(ss)

# Tracts where median age of the men is significantly different (equal to or greater than 20 years) from the median age of women Non-Hispanic White
hdftracts_males = dfhdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.len().alias('HDF_Males'))
mdftracts_males = dfmdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.len().alias('MDF_Males'))
hdftracts_females = dfhdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.len().alias('HDF_Females'))
mdftracts_females = dfmdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.len().alias('MDF_Females'))

hdftracts_gt5males = hdftracts_males.filter(pl.col('HDF_Males') >= 5).get_column('GEOID').to_list()
mdftracts_gt5males = mdftracts_males.filter(pl.col('MDF_Males') >= 5).get_column('GEOID').to_list()
hdftracts_gt5females = hdftracts_females.filter(pl.col('HDF_Females') >= 5).get_column('GEOID').to_list()
mdftracts_gt5females = mdftracts_females.filter(pl.col('MDF_Females') >= 5).get_column('GEOID').to_list()

hdftracts_gt5bothsex =  list(set(hdftracts_gt5males).intersection(hdftracts_gt5females))
mdftracts_gt5bothsex =  list(set(mdftracts_gt5males).intersection(mdftracts_gt5females))

hdftracts_gt5bothsex_index = hdftracts_gt5bothsex
mdftracts_gt5bothsex_index = mdftracts_gt5bothsex

hdftracts_malemedage = dfhdf().filter((pl.col('RACEALONE') == 1), (pl.col('CENHISP') == '1'), (pl.col('QSEX') == '1')).group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_MaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': hdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
mdftracts_malemedage = dfmdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '1').group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_MaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': mdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
hdftracts_femalemedage = dfhdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('HDF_FemaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': hdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)
mdftracts_femalemedage = dfmdf().filter((pl.col('RACEALONE') == 1)&(pl.col('CENHISP') == '1')&(pl.col('QSEX') == '2').group_by('TractGEOID').agg(pl.col('QAGE').map_elements(lambda x: median_grouped(x + 0.5), return_dtype=pl.Float64).alias('MDF_FemaleAge')).rename({'TractGEOID': 'GEOID'}).join(pl.DataFrame({'GEOID': mdftracts_gt5bothsex_index}), on='GEOID', how='right', coalesce=True)

hdftracts =  hdftracts_malemedage.join(hdftracts_femalemedage, on='GEOID', how="inner", coalesce=True)
mdftracts =  mdftracts_malemedage.join(mdftracts_femalemedage, on='GEOID', how="inner", coalesce=True)

hdftracts = hdftracts.with_columns(pl.when((pl.col('HDF_MaleAge').abs() - pl.col('HDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))
mdftracts = mdftracts.with_columns(pl.when((pl.col('MDF_MaleAge').abs() - pl.col('MDF_FemaleAge')) >= 20).then(1).otherwise(0).alias('BigAgeDiff'))

ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'HDF 20+ Year Median Age Diff Non-Hispanic White','NumCells':len(hdftracts), 'Inconsistent':hdftracts['BigAgeDiff'].sum()})
outputdflist.append(ss)
ss = pl.DataFrame({'Geography':'Tract', 'Size_Category':'All', 'Characteristic':'MDF 20+ Year Median Age Diff Non-Hispanic White','NumCells':len(mdftracts), 'Inconsistent':mdftracts['BigAgeDiff'].sum()})
outputdflist.append(ss)

# Output
outputdf = pl.concat(outputdflist, how='diagonal_relaxed')
outputdf.write_csv(f"{OUTPUTDIR}/cef_per_metrics_v2.csv")
print("{} All Done".format(datetime.now()))
