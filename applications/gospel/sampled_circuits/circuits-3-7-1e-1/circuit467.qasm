OPENQASM 2.0;
include "qelib1.inc";
qreg q468[3];
cx q468[2],q468[1];
rx(7*pi/4) q468[2];
cx q468[2],q468[1];
cx q468[1],q468[0];
