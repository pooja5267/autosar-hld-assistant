"""
Generate a synthetic AUTOSAR High-Level Design (HLD) PDF for educational testing.

This script creates a multi-page, realistic technical PDF specification for an
Electronic Braking and Stability Control System (EBSS).

DISCLAIMER: This document is fictional, synthetic, and intended solely for
educational benchmarking of the RAG assistant. It does not represent an official
AUTOSAR standard or proprietary automotive OEM design.
"""

import sys
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)

# Output directory configuration
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "sample_hld"
OUTPUT_FILE = OUTPUT_DIR / "Sample_AUTOSAR_BrakeSystem_HLD.pdf"


def create_sample_hld_pdf(output_path: Path = OUTPUT_FILE) -> Path:
    """Generate the synthetic 5-page AUTOSAR HLD PDF document."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Heading3"],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#333333"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        "DocCode",
        parent=styles["Code"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#742A2A"),
        backColor=colors.HexColor("#F7FAFC"),
        spaceAfter=4
    )

    disclaimer_style = ParagraphStyle(
        "DocDisclaimer",
        parent=styles["Italic"],
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#C53030"),
        spaceAfter=10
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, DOCUMENT CONTROL, EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("AUTOSAR High-Level Design (HLD) Document", title_style))
    story.append(Paragraph("System Module: Electronic Braking & Stability System (EBSS)", h1_style))
    story.append(Paragraph("Document Identifier: HLD-AUTOSAR-EBSS-2026-V2.1 | Classification: Synthetic Educational Artifact", body_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>IMPORTANT EDUCATIONAL DISCLAIMER:</b> This document contains purely synthetic, fictional "
        "automotive software architecture specifications created strictly for final-year engineering "
        "capstone demonstrations (PCCOE Capstone Project). It does NOT represent proprietary OEM designs "
        "or official AUTOSAR consortium standards.",
        disclaimer_style
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Document Control & Revision History", h1_style))
    rev_data = [
        ["Version", "Date", "Author", "Status", "Summary of Changes"],
        ["1.0", "2026-01-15", "System Architecture Team", "Released", "Initial baseline architecture for EBSS."],
        ["1.1", "2026-02-20", "Safety Integration Lead", "Approved", "Added failsafe redundancy metrics for pedal stroke sensors."],
        ["2.0", "2026-03-10", "Controls Engineering Group", "Approved", "Integrated Electronic Stability Control SWC and CAN message mappings."],
        ["2.1", "2026-04-05", "AUTOSAR RTE Team", "Current", "Refined client-server interfaces for MCAL valve block actuation."],
    ]
    rev_table = Table(rev_data, colWidths=[50, 70, 110, 60, 230])
    rev_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(rev_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("2. Executive Summary & Architectural Scope", h1_style))
    story.append(Paragraph(
        "The Electronic Braking & Stability System (EBSS) is an AUTOSAR Classic Release 4.4 compliant "
        "chassis domain subsystem. The primary objective of EBSS is to execute deterministic, real-time "
        "deceleration and yaw stabilization for light passenger vehicles. It abstracts sensor data, evaluates "
        "wheel slip ratios, integrates driver braking demands, and dispatches modulated hydraulic pressure commands "
        "to the electro-hydraulic brake actuator block.",
        body_style
    ))
    story.append(Paragraph(
        "The software architecture strictly adheres to layered AUTOSAR separation: the Application Layer "
        "comprises dedicated Software Components (SWCs), which interface seamlessly with the Runtime Environment (RTE) "
        "and communicate across Basic Software (BSW) stacks including MCAL, ComStack, and the OS.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: SYSTEM ARCHITECTURE & AUTOSAR LAYERING
    # =========================================================================
    story.append(Paragraph("3. System Architecture & AUTOSAR Layer Overview", h1_style))
    story.append(Paragraph(
        "The EBSS architecture is structured into three fundamental AUTOSAR operational tiers: "
        "the Application Layer, the Runtime Environment (RTE), and the Basic Software (BSW).",
        body_style
    ))

    story.append(Paragraph("3.1 AUTOSAR Layer Decomposition", h2_style))
    layer_data = [
        ["Layer", "Key Modules / Entities", "Primary Function & Execution Model"],
        ["Application Layer", "BrakePedalSensorSWC\nAntiLockBrakingSWC\nElectronicStabilityControlSWC\nBrakeActuatorDriverSWC", "Implements core braking algorithms, slip estimation, and driver intent calculation. Decoupled from hardware target."],
        ["Runtime Environment (RTE)", "Rte_EBSS_Generated_v2.1", "Enables inter-SWC Sender-Receiver and Client-Server communication. Manages event-driven and cyclic runnable triggers."],
        ["Base Software (BSW) Services", "Dem (Diag Event Mgr)\nFiM (Func Inhibit Mgr)\nNvM (Non-Volatile Mem)", "Handles fault recording, safety degradation inhibit conditions, and persistent calibration storage."],
        ["Communication Stack", "CanTp, CanIf, PduR, Com", "Marshals cyclic CAN frames (e.g. CAN_MSG_BRAKE_STATUS) over vehicle powertrain bus."],
        ["MCAL Drivers", "Adc, Pwm, Spi, Port, Dio", "Direct hardware abstraction for dual pedal potentiometer ADC channels and solenoid valve PWM actuation."],
    ]
    layer_table = Table(layer_data, colWidths=[110, 160, 250])
    layer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2C5282")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDF2F7")]),
    ]))
    story.append(layer_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("3.2 Hardware-to-Software Abstraction", h2_style))
    story.append(Paragraph(
        "Sensor signals enter the microcontroller via the Microcontroller Abstraction Layer (MCAL) ADC peripheral driver. "
        "The Sensor-Actuator SWCs package these physical ADC voltages into engineering units (% pedal travel, bar hydraulic pressure) "
        "and expose them across standardized Sender-Receiver interfaces to the core algorithmic Application SWCs.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: SOFTWARE COMPONENTS (SWCs) SPECIFICATION
    # =========================================================================
    story.append(Paragraph("4. Software Component (SWC) Specifications", h1_style))
    story.append(Paragraph(
        "This section details the functional responsibilities, internal runnable entities, and "
        "operational boundaries of each Application SWC.",
        body_style
    ))

    # SWC 1
    story.append(Paragraph("4.1 BrakePedalSensorSWC (Sensor-Actuator Component)", h2_style))
    story.append(Paragraph(
        "<b>Component Identifier:</b> SWC_BrakePedalSensor<br/>"
        "<b>Execution Period:</b> 5 ms cyclic runnable (Task_BrakeSensor_5ms)<br/>"
        "<b>Responsibilities:</b>",
        body_style
    ))
    story.append(Paragraph("• Reads raw dual potentiometer ADC voltages from channels ADC_CH_PEDAL_1 and ADC_CH_PEDAL_2 via MCAL.", bullet_style))
    story.append(Paragraph("• Computes cross-channel plausibility: if discrepancy between sensor 1 and sensor 2 exceeds 5.0% for > 40 ms, raises fault diagnostic DTC_PEDAL_PLAUSIBILITY_ERR.", bullet_style))
    story.append(Paragraph("• Converts verified analog stroke readings into normalized percentage signal Sig_PedalTravel_pct (0.0% to 100.0%).", bullet_style))
    story.append(Paragraph("• Categorizes driver deceleration urgency into Sig_BrakeIntent_enum (COMFORT_BRAKING, PANIC_BRAKING, STANDSTILL_HOLD).", bullet_style))
    story.append(Spacer(1, 6))

    # SWC 2
    story.append(Paragraph("4.2 AntiLockBrakingSWC (Algorithm SWC)", h2_style))
    story.append(Paragraph(
        "<b>Component Identifier:</b> SWC_AntiLockBraking<br/>"
        "<b>Execution Period:</b> 10 ms cyclic runnable (Task_ABS_10ms)<br/>"
        "<b>Responsibilities:</b>",
        body_style
    ))
    story.append(Paragraph("• Calculates longitudinal slip ratio for each individual wheel: FL, FR, RL, RR based on wheel speed pulses Sig_WheelSpeed_XX_kmh and vehicle reference speed.", bullet_style))
    story.append(Paragraph("• Detects incipient wheel lockup when wheel deceleration gradient exceeds -2.5g and slip ratio exceeds 18%.", bullet_style))
    story.append(Paragraph("• Triggers three-phase hydraulic modulation cycle (Pressure Hold, Pressure Decay, Pressure Re-apply) to maintain maximum tire-road friction.", bullet_style))
    story.append(Paragraph("• Publishes ABS active status flag Sig_ABS_Active_flg to the vehicle chassis coordinator over the RTE.", bullet_style))
    story.append(Spacer(1, 6))

    # SWC 3
    story.append(Paragraph("4.3 ElectronicStabilityControlSWC (Algorithm SWC)", h2_style))
    story.append(Paragraph(
        "<b>Component Identifier:</b> SWC_ElectronicStabilityControl<br/>"
        "<b>Execution Period:</b> 20 ms cyclic runnable (Task_ESC_20ms)<br/>"
        "<b>Responsibilities:</b>",
        body_style
    ))
    story.append(Paragraph("• Monitors lateral vehicle dynamics via inertial measurement unit signals: yaw rate Sig_YawRate_deg_s and lateral acceleration Sig_LateralAccel_g.", bullet_style))
    story.append(Paragraph("• Compares vehicle yaw response against standard Bicycle Dynamics Model derived from steering wheel angle.", bullet_style))
    story.append(Paragraph("• Mitigates understeer by applying targeted differential braking to inside rear wheel; mitigates oversteer by braking outside front wheel.", bullet_style))
    story.append(Paragraph("• Negotiates priority torque arbitration with Engine Management SWC to request powertrain throttle roll-back during severe skids.", bullet_style))
    story.append(Spacer(1, 6))

    # SWC 4
    story.append(Paragraph("4.4 BrakeActuatorDriverSWC (Actuator Component)", h2_style))
    story.append(Paragraph(
        "<b>Component Identifier:</b> SWC_BrakeActuatorDriver<br/>"
        "<b>Execution Period:</b> 5 ms cyclic runnable (Task_Actuator_5ms)<br/>"
        "<b>Responsibilities:</b>",
        body_style
    ))
    story.append(Paragraph("• Translates target hydraulic pressure demands (Sig_TargetPressure_FL_bar to RR) into calibrated duty cycles for inlet/outlet solenoid valves.", bullet_style))
    story.append(Paragraph("• Controls hydraulic pump motor brushless motor controller through SPI commands sent to MCAL Spi driver.", bullet_style))
    story.append(Paragraph("• Performs continuous coil resistance monitoring to detect open-circuit and short-to-ground conditions across all 8 hydraulic valves.", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: INTERFACES & SIGNAL DICTIONARY
    # =========================================================================
    story.append(Paragraph("5. Interface Catalog & Signal Dictionary", h1_style))
    story.append(Paragraph(
        "Inter-SWC communication is strictly managed through the RTE using either "
        "Sender-Receiver (S/R) data ports or Client-Server (C/S) synchronous service operations.",
        body_style
    ))

    story.append(Paragraph("5.1 AUTOSAR Interface Catalog", h2_style))
    if_data = [
        ["Interface Name", "Pattern", "Source SWC", "Target SWC(s)", "Description"],
        ["If_DriverBrakeRequest", "Sender-Receiver", "BrakePedalSensorSWC", "AntiLockBrakingSWC\nESC SWC", "Transmits calibrated pedal travel (%) and driver braking intent."],
        ["If_WheelSpeedData", "Sender-Receiver", "MCAL WheelSpeedIO", "AntiLockBrakingSWC\nESC SWC", "Delivers 4-wheel independent velocity signals."],
        ["If_StabilityIntervention", "Sender-Receiver", "ESC SWC", "AntiLockBrakingSWC\nEngineManagementSWC", "Dispatches differential torque/pressure corrections."],
        ["If_BrakePressureCommand", "Sender-Receiver", "AntiLockBrakingSWC", "BrakeActuatorDriverSWC", "Transmits target hydraulic pressures per caliper (bar)."],
        ["If_ActuatorHealthService", "Client-Server", "BrakeActuatorDriverSWC", "DiagnosticEventMgr", "Synchronous diagnostic routine to query valve coil electrical health."],
    ]
    if_table = Table(if_data, colWidths=[110, 75, 105, 105, 125])
    if_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2C5282")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(if_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("5.2 Signal Dictionary", h2_style))
    sig_data = [
        ["Signal Name", "Data Type", "Range / Units", "Interface", "Update Rate"],
        ["Sig_PedalTravel_pct", "float32", "0.0 .. 100.0 %", "If_DriverBrakeRequest", "5 ms"],
        ["Sig_BrakeIntent_enum", "uint8", "0=NONE, 1=COMFORT, 2=PANIC", "If_DriverBrakeRequest", "5 ms"],
        ["Sig_WheelSpeed_FL_kmh", "float32", "0.0 .. 350.0 km/h", "If_WheelSpeedData", "10 ms"],
        ["Sig_WheelSpeed_FR_kmh", "float32", "0.0 .. 350.0 km/h", "If_WheelSpeedData", "10 ms"],
        ["Sig_WheelSpeed_RL_kmh", "float32", "0.0 .. 350.0 km/h", "If_WheelSpeedData", "10 ms"],
        ["Sig_WheelSpeed_RR_kmh", "float32", "0.0 .. 350.0 km/h", "If_WheelSpeedData", "10 ms"],
        ["Sig_TargetPressure_FL_bar", "float32", "0.0 .. 250.0 bar", "If_BrakePressureCommand", "5 ms"],
        ["Sig_TargetPressure_FR_bar", "float32", "0.0 .. 250.0 bar", "If_BrakePressureCommand", "5 ms"],
        ["Sig_ABS_Active_flg", "boolean", "0=Inactive, 1=Active", "If_StabilityIntervention", "10 ms"],
        ["Sig_YawRate_deg_s", "float32", "-120.0 .. +120.0 deg/s", "If_StabilityIntervention", "20 ms"],
        ["Sig_LateralAccel_g", "float32", "-3.0 .. +3.0 g", "If_StabilityIntervention", "20 ms"],
    ]
    sig_table = Table(sig_data, colWidths=[125, 60, 120, 135, 80])
    sig_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDF2F7")]),
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("5.3 CAN Bus Frame Packaging", h2_style))
    story.append(Paragraph(
        "The external status frame <b>CAN_MSG_BRAKE_STATUS</b> is mapped to CAN ID <b>0x1A4</b> "
        "and broadcast cyclically every <b>10 ms</b> across the vehicle High-Speed CAN chassis bus. "
        "Payload includes Sig_ABS_Active_flg (bit 0), Sig_PedalTravel_pct (bytes 1-2, scaled x0.1), "
        "and primary system status bits.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: DEPENDENCIES, TIMING, AND SAFETY STATES
    # =========================================================================
    story.append(Paragraph("6. Inter-Component Dependencies & Timing Constraints", h1_style))
    story.append(Paragraph(
        "To ensure deterministic hard real-time execution in accordance with ISO 26262 ASIL-D requirements, "
        "strict data flow precedence and task execution cycles are enforced.",
        body_style
    ))

    story.append(Paragraph("6.1 Data Flow & SWC Interaction Hierarchy", h2_style))
    story.append(Paragraph(
        "1. <b>BrakePedalSensorSWC</b> executes first in the 5 ms cycle, sampling sensor inputs and updating the RTE buffer.<br/>"
        "2. <b>AntiLockBrakingSWC</b> depends directly on the output of BrakePedalSensorSWC and MCAL WheelSpeedIO. "
        "ABS calculations cannot proceed without fresh wheel speeds.<br/>"
        "3. <b>ElectronicStabilityControlSWC</b> arbitrates with AntiLockBrakingSWC. When ESC triggers an active yaw correction, "
        "its wheel braking demand supercedes standard ABS slip control on the stabilizing wheel.<br/>"
        "4. <b>BrakeActuatorDriverSWC</b> acts as the terminal sink component, receiving target pressures from ABS and ESC and commanding the hardware.",
        body_style
    ))

    story.append(Paragraph("6.2 Fault Handling & Safe States", h2_style))
    story.append(Paragraph("• <b>Dual Pedal Sensor Plausibility Failure:</b> If both stroke potentiometers disagree by > 5% for 40 ms, the system enters DEGRADED_PEDAL_FALLBACK mode. The regenerative braking request is cancelled and hydraulic backup is mandated.", bullet_style))
    story.append(Paragraph("• <b>Wheel Speed Loss:</b> If one wheel speed signal drops out, ABS function for that specific wheel is suppressed while retaining 3-wheel slip modulation.", bullet_style))
    story.append(Paragraph("• <b>Hydraulic Valve Driver Over-Temperature:</b> If actuator thermal sensor exceeds 135°C, high-frequency ABS pulsations are curtailed to protect MOSFET bridge drivers.", bullet_style))
    story.append(Paragraph("• <b>Complete System Failsafe State:</b> Upon critical RTE communication timeout (> 50 ms), the system cuts all solenoid power, causing valves to revert to mechanical fail-safe pass-through mode.", bullet_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("6.3 What is NOT Covered in this Document (Out of Scope)", h2_style))
    story.append(Paragraph(
        "This specification intentionally excludes: Long-range radar adaptive cruise control algorithms, "
        "electronic parking brake (EPB) cable puller actuators, high-voltage traction inverter regenerative blending, "
        "and steer-by-wire pinion actuators. These functions reside in separate chassis and ADAS HLD documents.",
        body_style
    ))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    out = create_sample_hld_pdf()
    print(f"Successfully generated synthetic AUTOSAR HLD PDF at: {out}")
