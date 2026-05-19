OPENQASM 2.0;
include "qelib1.inc";
qreg q435[3];
cx q435[1],q435[0];
cx q435[0],q435[1];
cx q435[1],q435[2];
cx q435[0],q435[1];
