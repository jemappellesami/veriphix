OPENQASM 2.0;
include "qelib1.inc";
qreg q84[6];
cx q84[4],q84[5];
cx q84[2],q84[1];
cx q84[3],q84[2];
cx q84[2],q84[1];
cx q84[0],q84[1];
