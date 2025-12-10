# Metadata Preservation: Quick Reference

## Example Subject: Thrx-CT001

### 📊 Summary Statistics

- **Original DICOM tags**: 24 (in this example file)
- **Preserved in BIDS JSON**: ~20-22 tags (83-92%)
- **Preserved in participants.tsv**: 4 fields (17%)
- **Discarded**: ~2-4 tags (8-17%)

---

## ✅ PRESERVED Metadata

### In BIDS Sidecar JSON (`.json` file)
✅ Patient demographics (anonymized)  
✅ Study dates and times  
✅ Series descriptions  
✅ Acquisition parameters (KVP, slice thickness, etc.)  
✅ Image geometry (pixel spacing, orientation)  
✅ Image properties (rows, columns, bits)  
✅ Equipment information (manufacturer, model)  
✅ Institution information  
✅ Intensity scaling (HU conversion)  

### In participants.tsv
✅ Participant ID (BIDS format: `sub-001`)  
✅ Original ID (`Thrx-CT001`)  
✅ Modality (`CT`)  
✅ Anatomy (`thorax`)  

---

## ❌ DISCARDED Metadata

### Instance-Level (per slice)
❌ `SOPInstanceUID` - Unique slice identifier  
❌ `InstanceNumber` - Slice number  
❌ `AcquisitionNumber` - Acquisition number  

### Privacy-Sensitive (usually removed)
❌ `PatientName` - Patient's name  
❌ Exact `PatientBirthDate` - Exact birth date  
❌ `ReferringPhysicianName` - Doctor's name  

### Redundant (can be computed)
❌ `SmallestPixelValueInSeries` - Can compute from NIfTI  
❌ `LargestPixelValueInSeries` - Can compute from NIfTI  
❌ `PixelDataGroupLength` - Not needed in NIfTI  

---

## 📋 Complete List by Category

### Patient Information
| Field | Preserved? | Location |
|-------|------------|----------|
| PatientID | ✅ Yes (anonymized) | BIDS JSON |
| PatientName | ❌ No (privacy) | Discarded |
| PatientSex | ✅ Yes | BIDS JSON |
| PatientAge | ✅ Yes | BIDS JSON |
| PatientBirthDate | ❌ No (privacy) | Discarded |

### Study Information
| Field | Preserved? | Location |
|-------|------------|----------|
| StudyDate | ✅ Yes | BIDS JSON |
| StudyTime | ✅ Yes | BIDS JSON |
| StudyDescription | ✅ Yes | BIDS JSON |
| StudyInstanceUID | ✅ Yes | BIDS JSON |
| StudyID | ✅ Yes | BIDS JSON |

### Series Information
| Field | Preserved? | Location |
|-------|------------|----------|
| SeriesDescription | ✅ Yes | BIDS JSON |
| SeriesNumber | ✅ Yes | BIDS JSON |
| SeriesInstanceUID | ✅ Yes | BIDS JSON |
| SeriesDate | ✅ Yes | BIDS JSON |
| SeriesTime | ✅ Yes | BIDS JSON |

### Acquisition Parameters
| Field | Preserved? | Location |
|-------|------------|----------|
| Modality | ✅ Yes | BIDS JSON + TSV |
| SliceThickness | ✅ Yes | BIDS JSON |
| PixelSpacing | ✅ Yes | BIDS JSON |
| KVP | ✅ Yes | BIDS JSON |
| TubeCurrent | ✅ Yes | BIDS JSON |
| ConvolutionKernel | ✅ Yes | BIDS JSON |

### Image Properties
| Field | Preserved? | Location |
|-------|------------|----------|
| Rows | ✅ Yes | BIDS JSON |
| Columns | ✅ Yes | BIDS JSON |
| BitsAllocated | ✅ Yes | BIDS JSON |
| BitsStored | ✅ Yes | BIDS JSON |
| PixelRepresentation | ✅ Yes | BIDS JSON |
| PhotometricInterpretation | ✅ Yes | BIDS JSON |

### Instance-Level (per slice)
| Field | Preserved? | Location |
|-------|------------|----------|
| SOPInstanceUID | ❌ No | Discarded |
| InstanceNumber | ❌ No | Discarded |
| AcquisitionNumber | ❌ No | Discarded |
| SliceLocation | ✅ Yes | BIDS JSON (as z-coord) |

### Equipment Information
| Field | Preserved? | Location |
|-------|------------|----------|
| Manufacturer | ✅ Yes | BIDS JSON |
| ManufacturerModelName | ✅ Yes | BIDS JSON |
| DeviceSerialNumber | ✅ Yes | BIDS JSON |
| StationName | ✅ Yes | BIDS JSON |
| SoftwareVersions | ✅ Yes | BIDS JSON |

---

## 💡 Key Takeaways

1. **Most metadata is preserved** (~85-90%) in BIDS sidecar JSON files
2. **Privacy-sensitive data is removed** (names, exact dates)
3. **Instance-level identifiers are discarded** (not needed in 3D volumes)
4. **Our pipeline extracts minimal metadata** (4 fields) - could be expanded
5. **BIDS JSON is the primary metadata source** - always keep these files!

---

## 📁 File Locations

- **Original DICOM**: `New_thorax_ct_dicom/Thrx-CT001/dicom/Thrx-CT001/image/*.dcm`
- **BIDS Sidecar JSON**: `data/bids_dataset/sub-001/anat/sub-001_CT.json` (if created)
- **participants.tsv**: `data/bids_dataset/participants.tsv`
- **NIfTI Image**: `data/bids_dataset/sub-001/anat/sub-001_CT.nii.gz`

