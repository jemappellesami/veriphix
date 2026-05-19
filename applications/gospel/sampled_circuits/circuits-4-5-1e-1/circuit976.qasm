OPENQASM 2.0;
include "qelib1.inc";
qreg q977[4];
rx(7*pi/4) q977[3];
rz(5*pi/4) q977[3];
cx q977[3],q977[2];
cx q977[1],q977[2];
cx q977[0],q977[1];
