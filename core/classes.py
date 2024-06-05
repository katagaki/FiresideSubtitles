from dataclasses import dataclass


@dataclass
class FiresideSegment:
    start: float
    end: float
    value: str


@dataclass
class FiresideFace:
    name: str
    x_start: int
    x_end: int
    y_start: int
    y_end: int


@dataclass
class FiresideFaceToSpeakerMapping:
    speaker_name: str
    person_name: str
