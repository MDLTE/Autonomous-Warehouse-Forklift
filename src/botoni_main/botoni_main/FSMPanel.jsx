/**
 * FSMPanel.jsx
 * ────────────
 * Panel visual de la máquina de estados para el dashboard.
 *
 * Props:
 *   ros  — instancia ROSLIB.Ros (del hook useRos)
 *
 * Muestra:
 *   - Estado actual (grande, con color)
 *   - Log del paso actual
 *   - Secuencia de estados (paso anterior / actual / siguiente)
 *   - Botones START / PAUSE / STOP
 *   - Selector de misión (LOADING → TRUCK / RACK → TRUCK)
 *   - Selector de empresa (activo solo en IDLE/STOP)
 */

import React from 'react';
import { useFSM, FSM_STATES, FSM_SEQUENCE } from './useFSM';

const MISSIONS = [
  { value: 'loading_truck', label: 'LOADING → TRUCK' },
  { value: 'rack_truck',    label: 'RACK → TRUCK'    },
];

const COMPANIES = ['EMEZON', 'POPSI', 'WALMAR'];

export default function FSMPanel({ ros }) {
  const {
    fsmState, fsmMission,
    isRunning, canStart, canPause, canStop,
    isPaused, isIdle, isError,
    sendStart, sendStop, sendPause,
    sendMission, sendCompany,
  } = useFSM(ros);

  const stateInfo = FSM_STATES[fsmState] || { label: fsmState, color: '#6b7280' };

  // Índice actual en la secuencia
  const currentIdx = FSM_SEQUENCE.indexOf(fsmState);
  const prevState  = currentIdx > 0  ? FSM_SEQUENCE[currentIdx - 1] : null;
  const nextState  = currentIdx >= 0 && currentIdx < FSM_SEQUENCE.length - 1
    ? FSM_SEQUENCE[currentIdx + 1] : null;

  return (
    <div style={styles.container}>

      {/* ── Estado actual ── */}
      <div style={{ ...styles.stateBox, borderColor: stateInfo.color }}>
        <div style={{ ...styles.stateDot, background: stateInfo.color }} />
        <span style={{ ...styles.stateLabel, color: stateInfo.color }}>
          {stateInfo.label}
        </span>
        {isPaused && (
          <span style={styles.pausedBadge}>⏸ PAUSADO</span>
        )}
        {isError && (
          <span style={styles.errorBadge}>⚠ ERROR</span>
        )}
      </div>

      {/* ── Log ── */}
      <div style={styles.logBox}>
        <span style={styles.logText}>{fsmMission.log}</span>
      </div>

      {/* ── Secuencia prev / current / next ── */}
      <div style={styles.sequenceRow}>
        <div style={styles.seqSlot}>
          {prevState && (
            <>
              <span style={styles.seqArrow}>▲</span>
              <span style={styles.seqLabel}>
                {FSM_STATES[prevState]?.label || prevState}
              </span>
            </>
          )}
        </div>
        <div style={{ ...styles.seqSlot, ...styles.seqCurrent,
                       borderColor: stateInfo.color }}>
          <span style={{ color: stateInfo.color, fontWeight: 700 }}>
            {stateInfo.label}
          </span>
        </div>
        <div style={styles.seqSlot}>
          {nextState && (
            <>
              <span style={styles.seqLabel}>
                {FSM_STATES[nextState]?.label || nextState}
              </span>
              <span style={styles.seqArrow}>▼</span>
            </>
          )}
        </div>
      </div>

      {/* ── Contexto de misión ── */}
      {(fsmMission.source_wp || fsmMission.dest_wp) && (
        <div style={styles.contextRow}>
          {fsmMission.source_wp && (
            <span style={styles.contextTag}>
              📦 {fsmMission.source_wp}
            </span>
          )}
          {fsmMission.dest_wp && (
            <span style={styles.contextTag}>
              🚛 {fsmMission.dest_wp}
            </span>
          )}
          {fsmMission.target && (
            <span style={{ ...styles.contextTag, background: '#1d4ed8' }}>
              🏷 {fsmMission.target}
            </span>
          )}
        </div>
      )}

      {/* ── Selector de misión (solo en IDLE/STOP) ── */}
      {(isIdle || fsmState === 'STOP') && (
        <div style={styles.selectorRow}>
          {MISSIONS.map((m) => (
            <button
              key={m.value}
              style={{
                ...styles.missionBtn,
                ...(fsmMission.mission === m.value ? styles.missionBtnActive : {}),
              }}
              onClick={() => sendMission(m.value)}
            >
              {m.label}
            </button>
          ))}
        </div>
      )}

      {/* ── Selector de empresa (solo en IDLE/STOP) ── */}
      {(isIdle || fsmState === 'STOP') && (
        <div style={styles.selectorRow}>
          {COMPANIES.map((co) => (
            <button
              key={co}
              style={{
                ...styles.companyBtn,
                ...(fsmMission.target === co ? styles.companyBtnActive : {}),
              }}
              onClick={() => sendCompany(co)}
            >
              {co}
            </button>
          ))}
        </div>
      )}

      {/* ── Botones de control ── */}
      <div style={styles.controlRow}>
        <button
          style={{ ...styles.ctrlBtn, ...styles.btnStop }}
          onClick={sendStop}
          disabled={!canStop}
        >
          ■ STOP
        </button>
        <button
          style={{ ...styles.ctrlBtn, ...styles.btnPause }}
          onClick={sendPause}
          disabled={!canPause}
        >
          ⏸ PAUSE
        </button>
        <button
          style={{ ...styles.ctrlBtn, ...styles.btnStart }}
          onClick={sendStart}
          disabled={!canStart}
        >
          {isPaused ? '▶ RESUME' : '▶ START'}
        </button>
      </div>

    </div>
  );
}

// ─── Estilos inline ──────────────────────────────────────────
const styles = {
  container: {
    background: '#0f172a',
    border: '1px solid #1e3a5f',
    borderRadius: 12,
    padding: 16,
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
    fontFamily: "'JetBrains Mono', 'Courier New', monospace",
    minWidth: 280,
  },
  stateBox: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '8px 12px',
    border: '2px solid',
    borderRadius: 8,
    background: '#0a1628',
  },
  stateDot: {
    width: 12,
    height: 12,
    borderRadius: '50%',
    flexShrink: 0,
    boxShadow: '0 0 8px currentColor',
  },
  stateLabel: {
    fontSize: 18,
    fontWeight: 700,
    letterSpacing: 2,
    flex: 1,
  },
  pausedBadge: {
    background: '#92400e',
    color: '#fbbf24',
    fontSize: 10,
    padding: '2px 6px',
    borderRadius: 4,
    fontWeight: 700,
  },
  errorBadge: {
    background: '#7f1d1d',
    color: '#fca5a5',
    fontSize: 10,
    padding: '2px 6px',
    borderRadius: 4,
    fontWeight: 700,
  },
  logBox: {
    background: '#0d1f2d',
    padding: '6px 10px',
    borderRadius: 6,
    borderLeft: '3px solid #1e3a5f',
  },
  logText: {
    color: '#94a3b8',
    fontSize: 12,
    lineHeight: 1.4,
  },
  sequenceRow: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
    alignItems: 'center',
  },
  seqSlot: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    color: '#4b5563',
    fontSize: 11,
    letterSpacing: 1,
    padding: '3px 0',
  },
  seqCurrent: {
    border: '1px solid',
    borderRadius: 6,
    padding: '4px 16px',
    background: '#0a1628',
    fontSize: 13,
  },
  seqArrow: {
    fontSize: 9,
    color: '#374151',
  },
  seqLabel: {
    color: '#6b7280',
  },
  contextRow: {
    display: 'flex',
    gap: 6,
    flexWrap: 'wrap',
  },
  contextTag: {
    background: '#1e3a5f',
    color: '#93c5fd',
    fontSize: 11,
    padding: '2px 8px',
    borderRadius: 4,
    fontFamily: 'monospace',
  },
  selectorRow: {
    display: 'flex',
    gap: 6,
    flexWrap: 'wrap',
  },
  missionBtn: {
    flex: 1,
    padding: '6px 8px',
    background: '#1e293b',
    border: '1px solid #334155',
    borderRadius: 6,
    color: '#94a3b8',
    fontSize: 11,
    cursor: 'pointer',
    fontFamily: 'inherit',
    letterSpacing: 1,
    transition: 'all 0.15s',
  },
  missionBtnActive: {
    background: '#1e3a5f',
    borderColor: '#3b82f6',
    color: '#60a5fa',
  },
  companyBtn: {
    flex: 1,
    padding: '6px 8px',
    background: '#1e293b',
    border: '1px solid #334155',
    borderRadius: 6,
    color: '#94a3b8',
    fontSize: 11,
    cursor: 'pointer',
    fontFamily: 'inherit',
    letterSpacing: 1,
    transition: 'all 0.15s',
  },
  companyBtnActive: {
    background: '#1e3a5f',
    borderColor: '#2563eb',
    color: '#93c5fd',
  },
  controlRow: {
    display: 'flex',
    gap: 8,
    marginTop: 4,
  },
  ctrlBtn: {
    flex: 1,
    padding: '10px 0',
    borderRadius: 8,
    border: 'none',
    cursor: 'pointer',
    fontSize: 13,
    fontWeight: 700,
    fontFamily: 'inherit',
    letterSpacing: 1,
    transition: 'opacity 0.15s, transform 0.1s',
  },
  btnStop: {
    background: '#7f1d1d',
    color: '#fca5a5',
  },
  btnPause: {
    background: '#78350f',
    color: '#fde68a',
  },
  btnStart: {
    background: '#14532d',
    color: '#86efac',
  },
};
