"""Sequence models over per-entity time series produced by an asset-pricing model.

Where :mod:`nekron.asset_pricing` explains the cross-section of returns at each
date, the models here read what is left over *along* time: one series per entity,
processed as fixed-length windows.
"""
