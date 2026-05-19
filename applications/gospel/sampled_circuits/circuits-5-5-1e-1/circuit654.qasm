OPENQASM 2.0;
include "qelib1.inc";
qreg q655[5];
rx(3*pi/2) q655[0];
cx q655[4],q655[3];
cx q655[0],q655[1];
cx q655[2],q655[1];
cx q655[3],q655[2];
