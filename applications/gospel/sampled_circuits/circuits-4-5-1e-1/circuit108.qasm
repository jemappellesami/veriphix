OPENQASM 2.0;
include "qelib1.inc";
qreg q109[4];
cx q109[1],q109[2];
rz(pi/2) q109[1];
cx q109[2],q109[3];
cx q109[2],q109[1];
cx q109[1],q109[0];
