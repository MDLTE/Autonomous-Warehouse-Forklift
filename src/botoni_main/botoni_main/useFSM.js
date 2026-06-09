/**
 * useFSM.js
 * ─────────
 * Hook React que conecta el dashboard con el FSM node.
 *
 * Suscribe a:
 *   /fsm_state    — string simple del estado
 *   /fsm_mission  — JSON completo {state, mission, target, log, source_wp, dest_wp}
 *
 * Publica en:
 *   /ui_cmd       — JSON {action, payload}
 *
 * Uso:
 *   const { fsmState, fsmMission, sendCmd, sendMission, sendCompany } = useFSM(ros);
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import ROSLIB from 'roslib';

// Todos los estados posibles de la FSM
export const FSM_STATES = {
  STOP:           { label: 'STOP',           color: '#ef4444', group: 'control'  },
  IDLE:           { label: 'IDLE',           color: '#6b7280', group: 'control'  },
  PAUSED:         { label: 'PAUSED',         color: '#f59e0b', group: 'control'  },
  ERROR:          { label: 'ERROR',          color: '#dc2626', group: 'control'  },
  SEARCH_LOADING: { label: 'BUSCAR LOADING', color: '#8b5cf6', group: 'search'   },
  SEARCH_RACK:    { label: 'BUSCAR RACK',    color: '#8b5cf6', group: 'search'   },
  NAV_TO_SOURCE:  { label: 'NAV → ORIGEN',   color: '#3b82f6', group: 'nav'      },
  ALIGN_SOURCE:   { label: 'ALINEAR ORIGEN', color: '#06b6d4', group: 'align'    },
  FORK_ENTER_GET: { label: 'FORK ENTRAR',    color: '#10b981', group: 'fork'     },
  FORK_GRAB_GET:  { label: 'FORK AGARRAR',   color: '#10b981', group: 'fork'     },
  REVERSE_GET:    { label: 'RETROCEDER',     color: '#84cc16', group: 'move'     },
  SEARCH_TRUCK:   { label: 'BUSCAR TRUCK',   color: '#8b5cf6', group: 'search'   },
  NAV_TO_DEST:    { label: 'NAV → DESTINO',  color: '#3b82f6', group: 'nav'      },
  ALIGN_DEST:     { label: 'ALINEAR DEST',   color: '#06b6d4', group: 'align'    },
  FORK_ENTER_PUT: { label: 'FORK ENTRAR',    color: '#f97316', group: 'fork'     },
  FORK_DROP_PUT:  { label: 'FORK DEPOSITAR', color: '#f97316', group: 'fork'     },
  REVERSE_PUT:    { label: 'RETROCEDER',     color: '#84cc16', group: 'move'     },
  MISSION_DONE:   { label: '✓ MISIÓN OK',    color: '#22c55e', group: 'done'     },
};

// Orden secuencial de los estados para visualización
export const FSM_SEQUENCE = [
  'STOP', 'IDLE',
  'SEARCH_LOADING', 'SEARCH_RACK',
  'NAV_TO_SOURCE',
  'ALIGN_SOURCE',
  'FORK_ENTER_GET', 'FORK_GRAB_GET', 'REVERSE_GET',
  'SEARCH_TRUCK',
  'NAV_TO_DEST',
  'ALIGN_DEST',
  'FORK_ENTER_PUT', 'FORK_DROP_PUT', 'REVERSE_PUT',
  'MISSION_DONE',
];

export function useFSM(ros) {
  const [fsmState, setFsmState]     = useState('STOP');
  const [fsmMission, setFsmMission] = useState({
    state: 'STOP',
    mission: 'none',
    target: null,
    log: 'Sistema listo.',
    source_wp: null,
    dest_wp: null,
    source_type: null,
  });

  const pubRef = useRef(null);

  // ── Setup de topics ────────────────────────────────────────
  useEffect(() => {
    if (!ros) return;

    // Subscriber /fsm_state
    const subState = new ROSLIB.Topic({
      ros,
      name: '/fsm_state',
      messageType: 'std_msgs/String',
    });
    subState.subscribe((msg) => {
      setFsmState(msg.data);
    });

    // Subscriber /fsm_mission
    const subMission = new ROSLIB.Topic({
      ros,
      name: '/fsm_mission',
      messageType: 'std_msgs/String',
    });
    subMission.subscribe((msg) => {
      try {
        const data = JSON.parse(msg.data);
        setFsmMission(data);
        if (data.state) setFsmState(data.state);
      } catch (e) {
        console.warn('[useFSM] Error parsing /fsm_mission:', e);
      }
    });

    // Publisher /ui_cmd
    pubRef.current = new ROSLIB.Topic({
      ros,
      name: '/ui_cmd',
      messageType: 'std_msgs/String',
    });

    return () => {
      subState.unsubscribe();
      subMission.unsubscribe();
    };
  }, [ros]);

  // ── Comandos de control ────────────────────────────────────
  const sendCmd = useCallback((action, payload = {}) => {
    if (!pubRef.current) return;
    pubRef.current.publish(
      new ROSLIB.Message({ data: JSON.stringify({ action, payload }) })
    );
    console.log(`[UI→FSM] ${action}`, payload);
  }, []);

  const sendStart   = useCallback(() => sendCmd('START'),              [sendCmd]);
  const sendStop    = useCallback(() => sendCmd('STOP'),               [sendCmd]);
  const sendPause   = useCallback(() => sendCmd('PAUSE'),              [sendCmd]);
  const sendMission = useCallback((mission) =>
    sendCmd('SELECT_MISSION', { mission }),                            [sendCmd]);
  const sendCompany = useCallback((company) =>
    sendCmd('SELECT_COMPANY', { company }),                            [sendCmd]);

  // ── Estado derivado ────────────────────────────────────────
  const isRunning  = !['STOP', 'IDLE', 'PAUSED', 'ERROR'].includes(fsmState);
  const canStart   = ['IDLE', 'PAUSED'].includes(fsmState);
  const canPause   = isRunning;
  const canStop    = fsmState !== 'STOP';
  const isPaused   = fsmState === 'PAUSED';
  const isIdle     = fsmState === 'IDLE';
  const isError    = fsmState === 'ERROR';

  return {
    fsmState,
    fsmMission,
    isRunning,
    canStart,
    canPause,
    canStop,
    isPaused,
    isIdle,
    isError,
    sendStart,
    sendStop,
    sendPause,
    sendMission,
    sendCompany,
    sendCmd,
  };
}
