OPENQASM 2.0;
include "qelib1.inc";
qreg q495[3];
rx(7*pi/4) q495[2];
cx q495[2],q495[1];
cx q495[1],q495[0];
