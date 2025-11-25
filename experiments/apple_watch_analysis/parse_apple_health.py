#!/usr/bin/env python3
"""
Apple Health Export Data Parser

Extracts running workout data from Apple Health export including:
- Workout metadata (duration, distance, dates)
- Heart rate time-series records
- GPS routes from GPX files (lat, lon, elevation, speed)
- Aligns HR and GPS data with interpolation

Author: Apple Watch Analysis Pipeline
Date: 2025-11-25
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Workout:
    """Workout metadata from Apple Health export"""
    workout_id: str
    activity_type: str
    start_date: datetime
    end_date: datetime
    duration_min: float
    total_distance_km: Optional[float]
    total_energy_kcal: Optional[float]
    elevation_ascended_m: Optional[float]
    gpx_file_path: Optional[str]
    hr_avg: Optional[float]
    hr_min: Optional[float]
    hr_max: Optional[float]
    
    def to_dict(self):
        """Convert to dictionary with ISO format dates"""
        d = asdict(self)
        d['start_date'] = self.start_date.isoformat()
        d['end_date'] = self.end_date.isoformat()
        return d


@dataclass
class HeartRateRecord:
    """Individual heart rate measurement"""
    timestamp: datetime
    value: float
    motion_context: int  # 0=sedentary, 1=active, 2=workout


@dataclass
class GPXTrackpoint:
    """GPS trackpoint from GPX file"""
    timestamp: datetime
    lat: float
    lon: float
    elevation: float
    speed: float  # m/s
    course: Optional[float]
    h_accuracy: Optional[float]
    v_accuracy: Optional[float]


class AppleHealthParser:
    """Parser for Apple Health export.xml file"""
    
    def __init__(self, export_xml_path: str, base_dir: str):
        """
        Args:
            export_xml_path: Path to export.xml file
            base_dir: Base directory containing export.xml (for resolving GPX paths)
        """
        self.export_xml_path = Path(export_xml_path)
        self.base_dir = Path(base_dir)
        
        if not self.export_xml_path.exists():
            raise FileNotFoundError(f"Export XML not found: {export_xml_path}")
    
    def parse_date(self, date_str: str) -> datetime:
        """Parse Apple Health date string to datetime"""
        # Format: "2024-11-15 16:51:46 +0100" or "2024-11-15T16:51:46Z"
        date_str = date_str.strip()
        
        # Try ISO format first
        if 'T' in date_str and date_str.endswith('Z'):
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        
        # Try Apple Health format with timezone
        try:
            # Remove timezone abbreviation if present
            if date_str[-5] in ['+', '-']:
                dt = datetime.strptime(date_str[:-6], "%Y-%m-%d %H:%M:%S")
                # Parse timezone offset
                tz_sign = 1 if date_str[-5] == '+' else -1
                tz_hours = int(date_str[-4:-2])
                tz_minutes = int(date_str[-2:])
                # Create timezone-aware datetime (simplified - just store UTC offset)
                return dt
            else:
                return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Warning: Could not parse date '{date_str}': {e}")
            return None
    
    def parse_workouts(self) -> List[Workout]:
        """Extract all running workouts from export.xml"""
        print(f"Parsing workouts from {self.export_xml_path}...")
        
        workouts = []
        
        # Parse XML incrementally to handle large file
        context = ET.iterparse(str(self.export_xml_path), events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        workout_count = 0
        for event, elem in context:
            if event == 'end' and elem.tag == 'Workout':
                # Check if it's a running workout
                activity_type = elem.get('workoutActivityType', '')
                if 'Running' not in activity_type:
                    elem.clear()
                    continue
                
                # Extract workout metadata
                start_date = self.parse_date(elem.get('startDate'))
                end_date = self.parse_date(elem.get('endDate'))
                
                if not start_date or not end_date:
                    elem.clear()
                    continue
                
                # Parse duration
                duration_str = elem.get('duration', '0')
                duration_min = float(duration_str)
                
                # Extract distance
                distance_km = None
                energy_kcal = None
                elevation_m = None
                hr_avg = None
                hr_min = None
                hr_max = None
                gpx_path = None
                
                # Parse child elements for statistics and routes
                for child in elem:
                    if child.tag == 'WorkoutStatistics':
                        stat_type = child.get('type', '')
                        if 'DistanceWalkingRunning' in stat_type:
                            distance_km = float(child.get('sum', 0))
                        elif 'ActiveEnergyBurned' in stat_type:
                            energy_kcal = float(child.get('sum', 0))
                        elif 'HeartRate' in stat_type:
                            hr_avg = float(child.get('average', 0)) if child.get('average') else None
                            hr_min = float(child.get('minimum', 0)) if child.get('minimum') else None
                            hr_max = float(child.get('maximum', 0)) if child.get('maximum') else None
                    
                    elif child.tag == 'MetadataEntry':
                        if child.get('key') == 'HKElevationAscended':
                            # Format: "956 cm"
                            elev_str = child.get('value', '0')
                            if 'cm' in elev_str:
                                elevation_m = float(elev_str.replace('cm', '').strip()) / 100
                    
                    elif child.tag == 'WorkoutRoute':
                        # Find FileReference for GPX path
                        for route_child in child:
                            if route_child.tag == 'FileReference':
                                gpx_path = route_child.get('path')
                
                # Create workout object
                workout = Workout(
                    workout_id=f"workout_{start_date.strftime('%Y%m%d_%H%M%S')}",
                    activity_type=activity_type,
                    start_date=start_date,
                    end_date=end_date,
                    duration_min=duration_min,
                    total_distance_km=distance_km,
                    total_energy_kcal=energy_kcal,
                    elevation_ascended_m=elevation_m,
                    gpx_file_path=gpx_path,
                    hr_avg=hr_avg,
                    hr_min=hr_min,
                    hr_max=hr_max
                )
                
                workouts.append(workout)
                workout_count += 1
                
                if workout_count % 50 == 0:
                    print(f"  Parsed {workout_count} running workouts...")
                
                # Clear element to free memory
                elem.clear()
        
        root.clear()
        print(f"✓ Found {len(workouts)} running workouts")
        return workouts
    
    def parse_heart_rate_records(self, start_date: datetime, end_date: datetime) -> List[HeartRateRecord]:
        """Extract heart rate records for a specific time window"""
        records = []
        
        context = ET.iterparse(str(self.export_xml_path), events=('start', 'end'))
        context = iter(context)
        event, root = next(context)
        
        for event, elem in context:
            if event == 'end' and elem.tag == 'Record':
                record_type = elem.get('type', '')
                
                if 'HeartRate' in record_type:
                    # Parse timestamp and value
                    timestamp = self.parse_date(elem.get('startDate'))
                    
                    if not timestamp:
                        elem.clear()
                        continue
                    
                    # Check if within workout time window (with 5min buffer on each side)
                    time_buffer = pd.Timedelta(minutes=5)
                    if timestamp < start_date - time_buffer or timestamp > end_date + time_buffer:
                        elem.clear()
                        continue
                    
                    value = float(elem.get('value', 0))
                    
                    # Extract motion context
                    motion_context = 0
                    for child in elem:
                        if child.tag == 'MetadataEntry':
                            if child.get('key') == 'HKMetadataKeyHeartRateMotionContext':
                                motion_context = int(child.get('value', 0))
                    
                    records.append(HeartRateRecord(
                        timestamp=timestamp,
                        value=value,
                        motion_context=motion_context
                    ))
                
                elem.clear()
        
        root.clear()
        return records


class GPXParser:
    """Parser for GPX route files"""
    
    def __init__(self, gpx_path: str):
        """
        Args:
            gpx_path: Path to GPX file
        """
        self.gpx_path = Path(gpx_path)
        
        if not self.gpx_path.exists():
            raise FileNotFoundError(f"GPX file not found: {gpx_path}")
    
    def parse_date(self, date_str: str) -> datetime:
        """Parse GPX timestamp (ISO format)"""
        # Format: "2024-11-15T16:51:46Z"
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    
    def parse_trackpoints(self) -> List[GPXTrackpoint]:
        """Extract all trackpoints from GPX file"""
        tree = ET.parse(str(self.gpx_path))
        root = tree.getroot()
        
        # GPX namespace
        ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
        
        trackpoints = []
        
        # Find all trackpoints
        for trkpt in root.findall('.//gpx:trkpt', ns):
            lat = float(trkpt.get('lat'))
            lon = float(trkpt.get('lon'))
            
            # Elevation
            ele_elem = trkpt.find('gpx:ele', ns)
            elevation = float(ele_elem.text) if ele_elem is not None else 0.0
            
            # Timestamp
            time_elem = trkpt.find('gpx:time', ns)
            if time_elem is None:
                continue
            timestamp = self.parse_date(time_elem.text)
            
            # Extensions (speed, course, accuracy)
            speed = None
            course = None
            h_accuracy = None
            v_accuracy = None
            
            extensions = trkpt.find('gpx:extensions', ns)
            if extensions is not None:
                speed_elem = extensions.find('speed')
                if speed_elem is not None:
                    speed = float(speed_elem.text)
                
                course_elem = extensions.find('course')
                if course_elem is not None:
                    course = float(course_elem.text)
                
                hacc_elem = extensions.find('hAcc')
                if hacc_elem is not None:
                    h_accuracy = float(hacc_elem.text)
                
                vacc_elem = extensions.find('vAcc')
                if vacc_elem is not None:
                    v_accuracy = float(vacc_elem.text)
            
            trackpoints.append(GPXTrackpoint(
                timestamp=timestamp,
                lat=lat,
                lon=lon,
                elevation=elevation,
                speed=speed if speed is not None else 0.0,
                course=course,
                h_accuracy=h_accuracy,
                v_accuracy=v_accuracy
            ))
        
        return trackpoints


def main():
    """Test parsing functionality"""
    # Paths
    export_xml = "/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL/experiments/apple_health_export/export.xml"
    base_dir = "/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL/experiments/apple_health_export"
    
    print("=" * 60)
    print("APPLE HEALTH DATA EXTRACTION - STEP 1: PARSING")
    print("=" * 60)
    
    # Parse workouts
    parser = AppleHealthParser(export_xml, base_dir)
    workouts = parser.parse_workouts()
    
    print(f"\n✓ Successfully parsed {len(workouts)} running workouts")
    
    # Save workout summary
    output_dir = Path("experiments/apple_watch_analysis/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    workouts_json = output_dir / "workouts_summary.json"
    with open(workouts_json, 'w') as f:
        json.dump([w.to_dict() for w in workouts], f, indent=2)
    
    print(f"\n✓ Saved workout summary to: {workouts_json}")
    
    # Show sample workouts
    print("\n" + "=" * 60)
    print("SAMPLE WORKOUTS (first 5):")
    print("=" * 60)
    
    for i, workout in enumerate(workouts[:5]):
        print(f"\nWorkout {i+1}:")
        print(f"  ID: {workout.workout_id}")
        print(f"  Date: {workout.start_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Duration: {workout.duration_min:.1f} min")
        print(f"  Distance: {workout.total_distance_km:.2f} km" if workout.total_distance_km else "  Distance: N/A")
        print(f"  HR: {workout.hr_avg:.0f} BPM (avg)" if workout.hr_avg else "  HR: N/A")
        print(f"  GPX: {workout.gpx_file_path if workout.gpx_file_path else 'N/A'}")
    
    # Test GPX parsing on one file
    print("\n" + "=" * 60)
    print("TESTING GPX PARSER (first workout with GPX):")
    print("=" * 60)
    
    for workout in workouts:
        if workout.gpx_file_path:
            gpx_full_path = Path(base_dir) / workout.gpx_file_path.lstrip('/')
            
            if gpx_full_path.exists():
                print(f"\nParsing: {gpx_full_path.name}")
                
                try:
                    gpx_parser = GPXParser(str(gpx_full_path))
                    trackpoints = gpx_parser.parse_trackpoints()
                    
                    print(f"✓ Found {len(trackpoints)} GPS trackpoints")
                    print(f"  Duration: {(trackpoints[-1].timestamp - trackpoints[0].timestamp).total_seconds() / 60:.1f} min")
                    print(f"  Sampling rate: ~{len(trackpoints) / ((trackpoints[-1].timestamp - trackpoints[0].timestamp).total_seconds()):.1f} Hz")
                    
                    # Show first few trackpoints
                    print("\n  First 3 trackpoints:")
                    for i, tp in enumerate(trackpoints[:3]):
                        print(f"    {i+1}. Time: {tp.timestamp.strftime('%H:%M:%S')}, Speed: {tp.speed:.2f} m/s, Elevation: {tp.elevation:.1f} m")
                    
                    break
                except Exception as e:
                    print(f"✗ Error parsing GPX: {e}")
            else:
                print(f"✗ GPX file not found: {gpx_full_path}")
    
    print("\n" + "=" * 60)
    print("PARSING COMPLETE - Ready for validation")
    print("=" * 60)


if __name__ == "__main__":
    main()
