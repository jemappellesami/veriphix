OPENQASM 2.0;
include "qelib1.inc";
qreg q794[4];
rx(7*pi/4) q794[3];
rz(5*pi/4) q794[3];
cx q794[3],q794[2];
cx q794[1],q794[2];
cx q794[1],q794[0];
