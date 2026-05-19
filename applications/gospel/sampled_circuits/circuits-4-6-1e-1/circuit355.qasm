OPENQASM 2.0;
include "qelib1.inc";
qreg q356[4];
cx q356[2],q356[1];
cx q356[2],q356[3];
cx q356[2],q356[1];
cx q356[1],q356[0];
rx(pi/4) q356[1];
