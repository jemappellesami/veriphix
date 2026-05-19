OPENQASM 2.0;
include "qelib1.inc";
qreg q109[7];
cx q109[2],q109[3];
cx q109[3],q109[4];
cx q109[2],q109[3];
cx q109[1],q109[2];
cx q109[0],q109[1];
