# Stage-2 V2 charging-power data card

## Frozen source and selection

The missing-power registry is derived from the National Laboratory of the
Rockies Alternative Fuels Data Center EV Charging Ports endpoint:

<https://developer.nlr.gov/docs/transportation/alt-fuel-stations-v1/ev-charging-units/>

The frozen source snapshot is
`data/sources/afdc/afdc_us_public_available_electric_ports.csv`, with SHA256
`bf3a0c796241b08ed862f650cf6ca299ada55661d5fe02d5c68fec8f28465ff6`.
The query scope is U.S., electric, public access, available status. Connector
compatibility is restricted to the reference vehicle: J1772 for AC Level 2
and CCS/J1772COMBO for DC fast charging.

## Statistic and result

The statistic is the connector-port-count-weighted national median of the
connector-specific power field. It is not a median of station-level maximums.

| Mode | Power field | Weight | Charging-unit observations | Connector-port weight | Frozen median |
| --- | --- | --- | ---: | ---: | ---: |
| AC Level 2 | `EV J1772 Power Output (kW)` | `EV J1772 Connector Count` | 149,719 | 149,738 | 6.5 kW |
| DC fast | `EV CCS Power Output (kW)` | `EV CCS Connector Count` | 36,211 | 37,052 | 200.0 kW |

The machine-readable result is
`configs/us_national_charging_power_medians_v1.json`, SHA256
`4a181a8440839602679be5f8d8c823b5d6f5fc5ff3b7211768c87d2fa0c87f57`.
The profile stores both the values and this registry hash. Slim export copies
the registry beside the profile, and restore verifies both hashes.

## Runtime semantics

Reported positive station power is preferred. If it is absent, the frozen
national median for that mode is required; absence of the median is a hard
generation error. The site/imputed value is capped by the reference vehicle,
then the canonical derating factor is applied:

```text
p_battery(q) = 0.90 * min(p_reported_or_national_median(q), p_vehicle_cap(q))
charge_time_s = delta_energy_kwh / p_battery_kw * 3600
```

Consequently the 200 kW CCS median is capped at 100 kW for this vehicle before
the 0.90 factor is applied. The 0.90 value is a frozen benchmark engineering
derating assumption, not an AFDC observation or a claim of charging
efficiency. Pilot sensitivity is reported at 0.85/0.90/0.95; canonical
artifacts store only 0.90.

The numerical precedent is documented, with its original semantics kept
explicit: NREL/TP-5400-89174 (2025) assumes 90% charger efficiency in a vehicle
life-cycle analysis, while NREL/TP-5400-71198 documents that EVI-Pro reduces
effective DC fast-charge rate for temperature and high-SOC tapering. This
benchmark uses the same 0.90 number as a transparent constant-power derating;
it does not claim that an efficiency loss, temperature effect, and charge
curve are physically identical.

- NREL/TP-5400-89174:
  <https://www.nrel.gov/docs/fy25osti/89174.pdf>
- NREL/TP-5400-71198:
  <https://www.nrel.gov/docs/fy19osti/71198.pdf>

## Limitations

- AFDC power completeness and reporting conventions vary by charging unit.
- A national median does not model local equipment mix or site power sharing.
- The benchmark uses infinite ports and does not model queueing.
- The linear full-charge model does not model a battery charging curve.
