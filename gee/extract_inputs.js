// AgriWatch locust module: extract input channels per 0.25-degree cell and 16-day step.
// Run in the Earth Engine Code Editor. Exports 8 CSV files to Google Drive (AgriWatch_locust).
//   MODIS MOD13Q1 v061 : ndvi, wi (water-sensitive index from NIR b02 and mid-IR b07), valid_frac
//   ERA5-Land daily    : t2m_c (mean 2 m temperature, deg C), precip_mm (16-day total), sm_l1 (mean soil water layer 1)
// Grid: lat 24-31N, lon 68-74E, 28 rows x 24 cols; row 0 = northernmost, col 0 = westernmost.
// Steps: MODIS-style 16-day composites (day-of-year 1, 17, ..., 353); last one is cut at 1 January.

var MODIS_SCALE = 500;     // metres; use 1000 if a MODIS task runs out of memory or time
var FIRST_YEAR = 2001, LAST_YEAR = 2020;

// ---------- grid of 672 cells
var cells = [];
for (var r = 0; r < 28; r++) {
  for (var c = 0; c < 24; c++) {
    var south = 24 + (27 - r) * 0.25, west = 68 + c * 0.25;
    cells.push(ee.Feature(
      ee.Geometry.Rectangle([west, south, west + 0.25, south + 0.25], null, false),
      {row: r, col: c, lat0: south, lon0: west}));
  }
}
var grid = ee.FeatureCollection(cells);

// ---------- 16-day step calendar
function fmt(ms) { return new Date(ms).toISOString().slice(0, 10); }
function stepsForYear(y) {
  var out = [], day = 86400000, jan1 = Date.UTC(y, 0, 1), next = Date.UTC(y + 1, 0, 1);
  for (var k = 0; k < 23; k++) {
    var s = jan1 + k * 16 * day;
    out.push([fmt(s), fmt(Math.min(s + 16 * day, next))]);
  }
  return out;
}

// ---------- one step, MODIS
function modisStep(startStr, endStr) {
  var s = ee.Date(startStr);
  var img = ee.Image(ee.ImageCollection('MODIS/061/MOD13Q1')
                       .filterDate(s, s.advance(1, 'day')).first());
  var ok = img.select('SummaryQA').lte(1);                       // good or marginal pixels
  var ndvi = img.select('NDVI').multiply(0.0001).updateMask(ok).rename('ndvi');
  var nir = img.select('sur_refl_b02').multiply(0.0001);
  var mir = img.select('sur_refl_b07').multiply(0.0001);
  var wi = nir.subtract(mir).divide(nir.add(mir)).updateMask(ok).rename('wi');
  var valid = ok.rename('valid_frac');                           // 1 = valid, 0 = not (unmasked)
  return ndvi.addBands(wi).addBands(valid)
    .reduceRegions({collection: grid, reducer: ee.Reducer.mean(),
                    scale: MODIS_SCALE, tileScale: 4})
    .map(function (f) { return f.set({step_start: startStr, step_end: endStr}); });
}

// ---------- one step, ERA5-Land
function eraStep(startStr, endStr) {
  var ic = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(startStr, endStr);
  var t = ic.select('temperature_2m').mean().subtract(273.15).rename('t2m_c');
  var p = ic.select('total_precipitation_sum').sum().multiply(1000).rename('precip_mm');
  var sm = ic.select('volumetric_soil_water_layer_1').mean().rename('sm_l1');
  var nDays = ic.size();
  return t.addBands(p).addBands(sm)
    .reduceRegions({collection: grid, reducer: ee.Reducer.mean(),
                    scale: 11132, tileScale: 2})
    .map(function (f) { return f.set({step_start: startStr, step_end: endStr, n_days: nDays}); });
}

// ---------- quick sanity test (prints 3 cells for one step)
print('MODIS test, 2019-08-13', modisStep('2019-08-13', '2019-08-29').limit(3));
print('ERA5-Land test, 2019-08-13', eraStep('2019-08-13', '2019-08-29').limit(3));

// ---------- exports: 4 five-year chunks x 2 datasets = 8 tasks
function exportRange(y0, y1) {
  var steps = [];
  for (var y = y0; y <= y1; y++) { steps = steps.concat(stepsForYear(y)); }
  var tag = y0 + '_' + y1;
  var modis = ee.FeatureCollection(ee.List(steps.map(function (s) { return modisStep(s[0], s[1]); }))).flatten();
  var era = ee.FeatureCollection(ee.List(steps.map(function (s) { return eraStep(s[0], s[1]); }))).flatten();
  Export.table.toDrive({
    collection: modis, description: 'locust_modis_' + tag, folder: 'AgriWatch_locust',
    fileNamePrefix: 'locust_modis_' + tag, fileFormat: 'CSV',
    selectors: ['step_start', 'step_end', 'row', 'col', 'lat0', 'lon0', 'ndvi', 'wi', 'valid_frac']});
  Export.table.toDrive({
    collection: era, description: 'locust_era5land_' + tag, folder: 'AgriWatch_locust',
    fileNamePrefix: 'locust_era5land_' + tag, fileFormat: 'CSV',
    selectors: ['step_start', 'step_end', 'row', 'col', 'lat0', 'lon0', 'n_days', 't2m_c', 'precip_mm', 'sm_l1']});
}
exportRange(2001, 2005);
exportRange(2006, 2010);
exportRange(2011, 2015);
exportRange(2016, 2020);
