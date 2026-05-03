from __future__ import annotations

from enum import Enum
from typing import Optional, Type

from pydantic import BaseModel, Field


class DocType(str, Enum):
    AADHAR = "aadhar"
    PAN_CARD = "pan_card"
    VOTER_ID = "voter_id"
    DRIVING_LICENSE = "driving_license"
    PASSPORT = "passport"
    RATION_CARD = "ration_card"
    LAND_RECORD = "land_record"
    BIRTH_CERTIFICATE = "birth_certificate"
    DEATH_CERTIFICATE = "death_certificate"
    MARRIAGE_CERTIFICATE = "marriage_certificate"
    CASTE_CERTIFICATE = "caste_certificate"
    INCOME_CERTIFICATE = "income_certificate"
    DOMICILE_CERTIFICATE = "domicile_certificate"
    AUTO = "auto"


class Member(BaseModel):
    name: Optional[str] = None
    relation: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None


class RationCard(BaseModel):
    card_number: Optional[str] = None
    card_type: Optional[str] = Field(None, description="APL/BPL/AAY/PHH etc.")
    head_of_family: Optional[str] = None
    address: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    issue_date: Optional[str] = Field(None, description="ISO 8601 if possible")
    expiry_date: Optional[str] = None
    members: list[Member] = Field(default_factory=list)


class LandRecord(BaseModel):
    survey_number: Optional[str] = None
    khasra_number: Optional[str] = None
    owner_name: Optional[str] = None
    fathers_name: Optional[str] = None
    village: Optional[str] = None
    tehsil: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    area: Optional[str] = Field(None, description="With units, e.g. '2.5 acres'")
    land_type: Optional[str] = Field(None, description="Agricultural / residential / commercial")
    mutation_date: Optional[str] = None


class BirthCertificate(BaseModel):
    registration_number: Optional[str] = None
    child_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    place_of_birth: Optional[str] = None
    gender: Optional[str] = None
    fathers_name: Optional[str] = None
    mothers_name: Optional[str] = None
    address: Optional[str] = None
    issue_date: Optional[str] = None
    issuing_authority: Optional[str] = None


class Aadhar(BaseModel):
    aadhar_number: Optional[str] = Field(None, description="12-digit UID, may be printed in 4-4-4 groups")
    name: Optional[str] = None
    date_of_birth: Optional[str] = Field(None, description="ISO 8601 if possible; year-only if just YOB is printed")
    gender: Optional[str] = None
    address: Optional[str] = None
    fathers_name: Optional[str] = Field(None, description="S/O, D/O, or W/O line if present")
    vid: Optional[str] = Field(None, description="16-digit Virtual ID, if printed")


class PANCard(BaseModel):
    pan_number: Optional[str] = Field(None, description="10-character alphanumeric, e.g. ABCDE1234F")
    name: Optional[str] = None
    fathers_name: Optional[str] = None
    date_of_birth: Optional[str] = None


class VoterID(BaseModel):
    epic_number: Optional[str] = Field(None, description="EPIC / Voter ID number")
    name: Optional[str] = None
    fathers_or_husbands_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    assembly_constituency: Optional[str] = None
    issue_date: Optional[str] = None


class DrivingLicense(BaseModel):
    license_number: Optional[str] = None
    name: Optional[str] = None
    fathers_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    address: Optional[str] = None
    blood_group: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    vehicle_classes: list[str] = Field(default_factory=list, description="e.g. ['LMV', 'MCWG']")
    issuing_authority: Optional[str] = None


class Passport(BaseModel):
    passport_number: Optional[str] = None
    surname: Optional[str] = None
    given_names: Optional[str] = None
    nationality: Optional[str] = None
    date_of_birth: Optional[str] = None
    place_of_birth: Optional[str] = None
    gender: Optional[str] = None
    date_of_issue: Optional[str] = None
    date_of_expiry: Optional[str] = None
    place_of_issue: Optional[str] = None
    fathers_name: Optional[str] = None
    mothers_name: Optional[str] = None
    spouse_name: Optional[str] = None
    address: Optional[str] = None


class DeathCertificate(BaseModel):
    registration_number: Optional[str] = None
    deceased_name: Optional[str] = None
    date_of_death: Optional[str] = None
    place_of_death: Optional[str] = None
    gender: Optional[str] = None
    age_at_death: Optional[str] = None
    fathers_or_spouses_name: Optional[str] = None
    address: Optional[str] = None
    cause_of_death: Optional[str] = None
    issue_date: Optional[str] = None
    issuing_authority: Optional[str] = None


class MarriageCertificate(BaseModel):
    registration_number: Optional[str] = None
    husband_name: Optional[str] = None
    wife_name: Optional[str] = None
    husband_age: Optional[int] = None
    wife_age: Optional[int] = None
    husband_address: Optional[str] = None
    wife_address: Optional[str] = None
    date_of_marriage: Optional[str] = None
    place_of_marriage: Optional[str] = None
    issue_date: Optional[str] = None
    issuing_authority: Optional[str] = None


class CasteCertificate(BaseModel):
    certificate_number: Optional[str] = None
    name: Optional[str] = None
    fathers_name: Optional[str] = None
    caste: Optional[str] = None
    sub_caste: Optional[str] = None
    category: Optional[str] = Field(None, description="SC / ST / OBC / EBC etc.")
    address: Optional[str] = None
    state: Optional[str] = None
    issue_date: Optional[str] = None
    issuing_authority: Optional[str] = None


class IncomeCertificate(BaseModel):
    certificate_number: Optional[str] = None
    name: Optional[str] = None
    fathers_or_husbands_name: Optional[str] = None
    annual_income: Optional[str] = Field(None, description="With currency/units, e.g. 'INR 2,40,000'")
    income_source: Optional[str] = None
    address: Optional[str] = None
    issue_date: Optional[str] = None
    valid_until: Optional[str] = None
    issuing_authority: Optional[str] = None


class DomicileCertificate(BaseModel):
    certificate_number: Optional[str] = None
    name: Optional[str] = None
    fathers_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    address: Optional[str] = None
    issue_date: Optional[str] = None
    issuing_authority: Optional[str] = None


SCHEMAS: dict[DocType, Type[BaseModel]] = {
    DocType.AADHAR: Aadhar,
    DocType.PAN_CARD: PANCard,
    DocType.VOTER_ID: VoterID,
    DocType.DRIVING_LICENSE: DrivingLicense,
    DocType.PASSPORT: Passport,
    DocType.RATION_CARD: RationCard,
    DocType.LAND_RECORD: LandRecord,
    DocType.BIRTH_CERTIFICATE: BirthCertificate,
    DocType.DEATH_CERTIFICATE: DeathCertificate,
    DocType.MARRIAGE_CERTIFICATE: MarriageCertificate,
    DocType.CASTE_CERTIFICATE: CasteCertificate,
    DocType.INCOME_CERTIFICATE: IncomeCertificate,
    DocType.DOMICILE_CERTIFICATE: DomicileCertificate,
}


def schema_for(doc_type: DocType) -> Type[BaseModel]:
    if doc_type == DocType.AUTO:
        raise ValueError("AUTO has no fixed schema; resolve before calling schema_for")
    return SCHEMAS[doc_type]
