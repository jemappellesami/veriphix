OPENQASM 2.0;
include "qelib1.inc";
qreg q93[3];
cx q93[1],q93[0];
rx(3*pi/4) q93[2];
rz(3*pi/4) q93[2];
rx(3*pi/2) q93[2];
cx q93[1],q93[2];
