OPENQASM 2.0;
include "qelib1.inc";
qreg q470[3];
rx(5*pi/4) q470[0];
cx q470[1],q470[0];
cx q470[2],q470[1];
rx(pi/4) q470[0];
