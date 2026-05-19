OPENQASM 2.0;
include "qelib1.inc";
qreg q443[4];
rz(5*pi/4) q443[0];
cx q443[1],q443[0];
cx q443[0],q443[1];
cx q443[2],q443[1];
cx q443[1],q443[0];
