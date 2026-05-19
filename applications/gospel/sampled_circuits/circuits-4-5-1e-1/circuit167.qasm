OPENQASM 2.0;
include "qelib1.inc";
qreg q168[4];
rx(5*pi/4) q168[0];
cx q168[1],q168[0];
cx q168[2],q168[1];
cx q168[3],q168[2];
